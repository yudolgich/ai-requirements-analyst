from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pytest

from graph import build_analysis_graph
import graph
from gemini import GeminiRequestError, ResilientGeminiChat
from memory import AnalysisMemory
from mcp_tools import FilesystemMCPBridge
from rag import KnowledgeBase


class RecordingModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, messages: list[object]) -> AIMessage:
        prompt = str(messages[1].content)
        self.prompts.append(prompt)
        return AIMessage(content=f"Результат агента {len(self.prompts)}")


class StaticKnowledge:
    def search(self, query: str) -> list[dict[str, str]]:
        return [{"source": "fixture.md", "text": "Требование должно быть проверяемым."}]


class StaticMCP:
    def fetch_knowledge_context(self) -> dict[str, object]:
        return {
            "context": "Материал, прочитанный через MCP.",
            "sources": ["fixture.md"],
            "status": "MCP подключен и прочитал файл.",
        }


def test_agents_run_sequentially_and_receive_previous_results() -> None:
    model = RecordingModel()
    graph = build_analysis_graph(model=model, rag_store=StaticKnowledge(), mcp_bridge=StaticMCP())

    result = graph.invoke({"requirement": "Восстановление пароля по электронной почте"})

    assert [result[key] for key in ("analysis", "clarification", "formalization", "quality")] == [
        f"Результат агента {index}" for index in range(1, 5)
    ]
    assert len(model.prompts) == 4
    assert "Результат агента 1" in model.prompts[1]
    assert "Результат агента 2" in model.prompts[2]
    assert "Результат агента 3" in model.prompts[3]
    assert "Требование должно быть проверяемым" in model.prompts[0]
    assert "Материал, прочитанный через MCP" in model.prompts[0]
    assert result["rag_sources"] == ["fixture.md"]


def test_history_survives_reopening_database(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    memory = AnalysisMemory(database)
    session_id = memory.create_session()
    memory.save_analysis(
        session_id,
        "Восстановление пароля",
        {
            "analysis": "Есть неуточненный срок действия ссылки.",
            "clarification": "Как долго действует ссылка?",
            "formalization": "Пользователь восстанавливает пароль.",
            "quality": "Уточнить срок ссылки.",
            "rag_sources": ["fixture.md"],
            "rag_status": "Найдено фрагментов: 1",
            "mcp_sources": ["guide.md"],
            "mcp_status": "MCP подключен.",
        },
    )

    reopened = AnalysisMemory(database)
    history = reopened.get_history(session_id)

    assert history[0]["requirement"] == "Восстановление пароля"
    assert "Пользователь восстанавливает пароль" in reopened.build_prior_context(session_id)
    assert reopened.list_sessions()[0]["analyses"] == 1
    assert len(reopened.list_sessions(limit=None)) == 1
    assert history[0]["results"] == {
        "analysis": "Есть неуточненный срок действия ссылки.",
        "clarification": "Как долго действует ссылка?",
        "formalization": "Пользователь восстанавливает пароль.",
        "quality": "Уточнить срок ссылки.",
        "rag_sources": ["fixture.md"],
        "rag_status": "Найдено фрагментов: 1",
        "mcp_sources": ["guide.md"],
        "mcp_status": "MCP подключен.",
    }


def test_graph_stream_reports_sequential_stages_and_persists_checkpoint(
    tmp_path: Path,
) -> None:
    model = RecordingModel()
    config = {"configurable": {"thread_id": "stream-test-session"}}
    with SqliteSaver.from_conn_string(str(tmp_path / "stream.sqlite3")) as checkpointer:
        analysis_graph = build_analysis_graph(
            model=model,
            rag_store=StaticKnowledge(),
            mcp_bridge=StaticMCP(),
            checkpointer=checkpointer,
        )
        events = list(
            analysis_graph.stream(
                {"requirement": "Пароль сбрасывается"},
                config=config,
                stream_mode="updates",
            )
        )
        saved_state = analysis_graph.get_state(config).values

    assert [next(iter(event)) for event in events] == [
        "mcp_context",
        "rag_retrieval",
        "requirements_analyst",
        "clarification_specialist",
        "requirements_formalizer",
        "quality_controller",
    ]
    assert len(model.prompts) == 4
    assert saved_state["quality"] == "Результат агента 4"


def test_retry_after_agent_failure_resumes_checkpoint_without_repeating_completed_agent(
    tmp_path: Path,
) -> None:
    from google.genai.errors import ServerError

    class FailClarifierOnce:
        def __init__(self) -> None:
            self.calls: dict[str, int] = {}

        def invoke(self, messages: list[object]) -> AIMessage:
            role = str(messages[0].content)
            agent = role.split(". ", 1)[0]
            self.calls[agent] = self.calls.get(agent, 0) + 1
            if "уточнению требований" in role and self.calls[agent] == 1:
                raise ServerError(503, {"message": "temporary overload"})
            return AIMessage(content=f"Результат {agent}")

    model = FailClarifierOnce()
    config = {"configurable": {"thread_id": "retry-checkpoint-session"}}
    input_state = {"requirement": "Пароль сбрасывается"}
    checkpoint_path = tmp_path / "retry.sqlite3"
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        analysis_graph = build_analysis_graph(model=model, checkpointer=checkpointer)
        with pytest.raises(ServerError):
            list(analysis_graph.stream(input_state, config=config, stream_mode="updates"))

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        analysis_graph = build_analysis_graph(model=model, checkpointer=checkpointer)
        events = list(
            analysis_graph.stream(None, config=config, stream_mode="updates")
        )
        saved_state = analysis_graph.get_state(config).values

    resumed_nodes = [next(iter(event)) for event in events]
    assert "mcp_context" not in resumed_nodes
    assert "rag_retrieval" not in resumed_nodes
    assert "requirements_analyst" not in resumed_nodes
    assert model.calls["Ты специализированный агент: аналитик требований"] == 1
    assert model.calls["Ты специализированный агент: специалист по уточнению требований"] == 2
    assert model.calls["Ты специализированный агент: специалист по формализации требований"] == 1
    assert model.calls["Ты специализированный агент: контролер качества требований"] == 1
    assert saved_state["analysis"] == "Результат Ты специализированный агент: аналитик требований"
    assert saved_state["quality"] == "Результат Ты специализированный агент: контролер качества требований"


def test_analysis_memory_updates_partial_record_until_completion(tmp_path: Path) -> None:
    memory = AnalysisMemory(tmp_path / "partial.sqlite3")
    session_id = memory.create_session()
    partial_state = {
        "analysis": "Аналитик завершил.",
        "rag_status": "Найдено 2 фрагмента.",
        "rag_sources": ["guide.md"],
    }

    analysis_id = memory.save_analysis(
        session_id,
        "Тестовое требование",
        partial_state,
        status="in_progress",
    )
    memory.save_analysis(
        session_id,
        "Тестовое требование",
        partial_state,
        status="failed",
        analysis_id=analysis_id,
        error_message="Временная ошибка Gemini.",
    )
    history_after_failure = memory.get_history(session_id)

    assert len(history_after_failure) == 1
    assert history_after_failure[0]["status"] == "failed"
    assert history_after_failure[0]["error_message"] == "Временная ошибка Gemini."
    assert history_after_failure[0]["results"]["analysis"] == "Аналитик завершил."
    assert memory.build_prior_context(session_id) == ""

    memory.save_analysis(
        session_id,
        "Тестовое требование",
        {**partial_state, "quality": "Проверка завершена."},
        status="completed",
        analysis_id=analysis_id,
    )
    completed_history = memory.get_history(session_id)
    assert len(completed_history) == 1
    assert completed_history[0]["status"] == "completed"
    assert "Проверка завершена" in memory.build_prior_context(session_id)


def test_analysis_submission_claim_blocks_duplicate_rerun_and_completed_history(
    tmp_path: Path,
) -> None:
    from web_app import _claim_analysis_submission

    state: dict[str, object] = {}
    memory = AnalysisMemory(tmp_path / "idempotency.sqlite3")
    session_id = memory.create_session()
    empty_history = memory.get_history(session_id)

    assert _claim_analysis_submission(state, session_id, "Повторный запрос", empty_history)
    assert not _claim_analysis_submission(state, session_id, "Повторный запрос", empty_history)

    memory.save_analysis(
        session_id,
        "Сохранённое требование",
        {"analysis": "Готово."},
        status="completed",
    )
    assert not _claim_analysis_submission(
        {},
        session_id,
        "Сохранённое требование",
        memory.get_history(session_id),
    )


def test_chroma_indexes_and_retrieves_document(tmp_path: Path) -> None:
    documents = tmp_path / "knowledge"
    documents.mkdir()
    (documents / "test-guide.md").write_text(
        "Демонстрационный материал. Критерии приемки должны быть проверяемыми.",
        encoding="utf-8",
    )
    knowledge = KnowledgeBase(documents, tmp_path / "chroma")

    assert knowledge.index_documents() > 0
    results = knowledge.search("проверяемые критерии приемки")

    assert results
    assert results[0]["source"] == "test-guide.md"
    assert "проверяемыми" in results[0]["text"]


def test_sqlite_checkpoint_survives_graph_reopen(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "persistent-test-session"}}
    model = RecordingModel()
    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        graph = build_analysis_graph(model=model, checkpointer=checkpointer)
        graph.invoke({"requirement": "Пароль сбрасывается"}, config=config)

    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        graph = build_analysis_graph(model=RecordingModel(), checkpointer=checkpointer)
        saved_state = graph.get_state(config).values

    assert saved_state["analysis"] == "Результат агента 1"
    assert saved_state["formalization"] == "Результат агента 3"


def test_rag_failure_does_not_stop_analysis() -> None:
    class BrokenKnowledge:
        def search(self, query: str) -> list[dict[str, str]]:
            raise OSError("index unavailable")

    result = build_analysis_graph(
        model=RecordingModel(), rag_store=BrokenKnowledge()
    ).invoke({"requirement": "Пользователь восстанавливает пароль"})

    assert result["analysis"] == "Результат агента 1"
    assert "недоступен" in result["rag_status"]


def test_missing_mcp_server_is_reported_without_crashing(tmp_path: Path) -> None:
    def unavailable_server(config: dict[str, object]) -> object:
        raise FileNotFoundError("MCP executable unavailable")

    result = FilesystemMCPBridge(
        tmp_path, client_factory=unavailable_server
    ).fetch_knowledge_context()

    assert result["context"] == ""
    assert "MCP недоступен" in result["status"]
    assert "FileNotFoundError" in result["status"]


def test_gemini_retries_transient_failure_on_same_catalogued_model() -> None:
    calls: list[str] = []

    class FakeClient:
        def __init__(self, name: str) -> None:
            self.name = name

        def invoke(self, messages: list[object], **kwargs: object) -> AIMessage:
            calls.append(self.name)
            if len(calls) == 1:
                from google.genai.errors import ServerError

                raise ServerError(503, {"message": "temporary overload"})
            return AIMessage(content="Ответ после повтора")

    model = ResilientGeminiChat(
        api_key="test-only",
        preferred_model="gemini-3.8-flash",
        retry_budget=10,
        max_attempts=2,
        catalog_loader=lambda key, timeout: {
            "gemini-3.8-flash": {"generateContent"},
            "gemini-3.6-flash": {"generateContent"},
        },
        model_factory=lambda name, timeout: FakeClient(name),
        sleep=lambda delay: None,
        jitter=lambda low, high: 0,
    )

    response = model.invoke(["test"])

    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash"]
    assert response.content == "Ответ после повтора"
    assert model.model_name == "gemini-3.8-flash"


@pytest.mark.parametrize(
    ("status_code", "error_type", "category"),
    [
        (503, "server", "temporary"),
        (504, "server", "temporary"),
        (429, "server", "quota"),
        (401, "client", "authorization"),
        (403, "client", "authorization"),
        (400, "client", "configuration"),
        (400, "invalid-key", "authorization"),
    ],
)
def test_gemini_classifies_api_errors_and_retries_only_retryable_statuses(
    status_code: int,
    error_type: str,
    category: str,
) -> None:
    from google.genai.errors import ClientError, ServerError

    calls: list[str] = []
    delays: list[float] = []

    class FailingClient:
        def invoke(self, messages: list[object], **kwargs: object) -> AIMessage:
            calls.append("invoke")
            if error_type == "invalid-key":
                raise ClientError(
                    status_code,
                    {"message": "API key not valid. Please pass a valid API key."},
                )
            error_class = ServerError if error_type == "server" else ClientError
            raise error_class(status_code, {"message": "synthetic test failure"})

    model = ResilientGeminiChat(
        api_key="test-key",
        preferred_model="gemini-3.8-flash",
        retry_budget=30,
        max_attempts=3,
        catalog_loader=lambda key, timeout: {
            "gemini-3.8-flash": {"generateContent"},
            "gemini-3.9-flash": {"generateContent"},
        },
        model_factory=lambda name, timeout: FailingClient(),
        sleep=delays.append,
        jitter=lambda low, high: high,
    )

    with pytest.raises(GeminiRequestError) as raised:
        model.invoke(["test"])

    assert raised.value.category == category
    expected_calls = 1 if category in {"authorization", "configuration"} else 3
    assert len(calls) == expected_calls
    assert len(delays) == expected_calls - 1
    assert "gemini-3.9-flash" not in str(raised.value)


def test_gemini_uses_configured_or_catalog_verified_generate_content_model() -> None:
    created_models: list[str] = []

    class SuccessfulClient:
        def invoke(self, messages: list[object], **kwargs: object) -> AIMessage:
            return AIMessage(content="ok")

    model = ResilientGeminiChat(
        api_key="test-key",
        preferred_model="gemini-not-listed-flash",
        catalog_loader=lambda key, timeout: {
            "gemini-4.1-flash": {"generateContent"},
            "gemini-4.2-flash": {"embedContent"},
            "gemini-3.8-flash": {"generateContent"},
        },
        model_factory=lambda name, timeout: (created_models.append(name) or SuccessfulClient()),
    )

    model.invoke(["test"])

    assert created_models == ["gemini-3.8-flash"]


def test_gemini_respects_configured_model_when_catalog_confirms_generate_content() -> None:
    created_models: list[str] = []

    class SuccessfulClient:
        def invoke(self, messages: list[object], **kwargs: object) -> AIMessage:
            return AIMessage(content="ok")

    model = ResilientGeminiChat(
        api_key="test-key",
        preferred_model="gemini-2.5-flash",
        catalog_loader=lambda key, timeout: {
            "gemini-2.5-flash": {"generateContent"},
            "gemini-4.2-flash": {"generateContent"},
        },
        model_factory=lambda name, timeout: (created_models.append(name) or SuccessfulClient()),
    )

    model.invoke(["test"])

    assert created_models == ["gemini-2.5-flash"]


def test_model_catalog_selection_uses_preference_or_highest_verified_model() -> None:
    from gemini import select_catalog_model

    catalog = {
        "gemini-3.8-flash": {"generateContent"},
        "gemini-4.1-flash": {"generateContent"},
        "gemini-9.9-flash": {"embedContent"},
    }

    assert select_catalog_model(catalog, "gemini-3.8-flash") == "gemini-3.8-flash"
    assert select_catalog_model(catalog) == "gemini-3.8-flash"
    assert select_catalog_model(catalog, "not-in-catalog") == "gemini-3.8-flash"
    with pytest.raises(GeminiRequestError, match="Укажите GEMINI_MODEL явно"):
        select_catalog_model({"gemini-4.1-flash": {"generateContent"}})


def test_probe_sends_exactly_one_request_to_catalog_verified_model() -> None:
    import gemini

    calls: list[dict[str, object]] = []

    class FakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            calls.append({"messages": messages})
            return AIMessage(content="OK")

    def model_factory(name: str, api_key: str, timeout: float) -> FakeModel:
        calls.append({"name": name, "api_key": api_key, "timeout": timeout})
        return FakeModel()

    response = gemini.probe_text_generation_model(
        "test-key",
        "models/gemini-4.1-flash",
        {"gemini-4.1-flash": {"generateContent"}},
        timeout_seconds=10,
        model_factory=model_factory,
    )

    assert response.content == "OK"
    assert len(calls) == 2
    assert calls[0] == {
        "name": "gemini-4.1-flash",
        "api_key": "test-key",
        "timeout": 10,
    }


def test_probe_rejects_model_not_confirmed_by_catalog() -> None:
    from gemini import probe_text_generation_model

    with pytest.raises(GeminiRequestError, match="не подтверждена каталогом"):
        probe_text_generation_model(
            "test-key",
            "gemini-unverified-flash",
            {"gemini-4.1-flash": {"generateContent"}},
            model_factory=lambda name, key, timeout: None,
        )


def test_model_catalog_loads_only_models_advertising_generate_content(monkeypatch) -> None:
    from types import SimpleNamespace
    import gemini

    captured: dict[str, object] = {}

    class FakeModels:
        def list(self, config: dict[str, int]) -> list[SimpleNamespace]:
            captured["page_size"] = config["page_size"]
            return [
                SimpleNamespace(
                    name="models/gemini-3.8-flash",
                    supported_actions=["generateContent", "countTokens"],
                ),
                SimpleNamespace(name="models/embed-model", supported_actions=["embedContent"]),
                SimpleNamespace(name=None, supported_actions=["generateContent"]),
            ]

    def fake_client(*, api_key: str, http_options: object) -> SimpleNamespace:
        captured["api_key"] = api_key
        captured["timeout_ms"] = http_options.timeout
        return SimpleNamespace(models=FakeModels())

    monkeypatch.setattr(gemini.genai, "Client", fake_client)

    catalog = gemini.load_text_generation_models("test-key", 2.5)

    assert catalog == {
        "gemini-3.8-flash": {"generateContent", "countTokens"},
    }
    assert captured == {
        "api_key": "test-key",
        "timeout_ms": 2500,
        "page_size": 1000,
    }


def test_probe_sdk_client_disables_retries_and_honors_short_timeout(monkeypatch) -> None:
    import gemini

    captured_options: dict[str, object] = {}

    class FakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            return AIMessage(content="OK")

    def fake_chat_model(**kwargs: object) -> FakeModel:
        captured_options.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(gemini, "ChatGoogleGenerativeAI", fake_chat_model)

    gemini.probe_text_generation_model(
        "test-key",
        "gemini-4.1-flash",
        {"gemini-4.1-flash": {"generateContent"}},
        timeout_seconds=10,
    )

    assert captured_options["request_timeout"] == 10
    assert captured_options["retries"] == 0
    assert captured_options["model"] == "gemini-4.1-flash"


def test_probe_rejects_timeout_shorter_than_gemini_sdk_minimum() -> None:
    from gemini import probe_text_generation_model

    with pytest.raises(ValueError, match="at least 10 seconds"):
        probe_text_generation_model(
            "test-key",
            "gemini-3.8-flash",
            {"gemini-3.8-flash": {"generateContent"}},
            timeout_seconds=8,
            model_factory=lambda name, key, timeout: None,
        )


def test_gemini_retries_transient_model_catalog_failure_within_same_budget() -> None:
    from google.genai.errors import ServerError

    catalog_calls = 0
    delays: list[float] = []

    def load_catalog(key: str, timeout: float) -> dict[str, set[str]]:
        nonlocal catalog_calls
        catalog_calls += 1
        if catalog_calls == 1:
            raise ServerError(503, {"message": "temporary catalog overload"})
        return {"gemini-3.8-flash": {"generateContent"}}

    class SuccessfulClient:
        def invoke(self, messages: list[object], **kwargs: object) -> AIMessage:
            return AIMessage(content="ok")

    model = ResilientGeminiChat(
        api_key="test-key",
        preferred_model="gemini-3.8-flash",
        retry_budget=10,
        catalog_loader=load_catalog,
        model_factory=lambda name, timeout: SuccessfulClient(),
        sleep=delays.append,
        jitter=lambda low, high: high,
    )

    response = model.invoke(["test"])

    assert response.content == "ok"
    assert catalog_calls == 2
    assert delays == [2.0]


def test_gemini_redacts_api_key_from_retry_logs_and_errors(caplog) -> None:
    from google.genai.errors import ServerError

    secret = "test-api-key-secret"

    class FailedClient:
        def invoke(self, messages: list[object], **kwargs: object) -> AIMessage:
            raise ServerError(503, {"message": f"failed with {secret}"})

    model = ResilientGeminiChat(
        api_key=secret,
        preferred_model="gemini-3.8-flash",
        retry_budget=10,
        max_attempts=1,
        catalog_loader=lambda key, timeout: {"gemini-3.8-flash": {"generateContent"}},
        model_factory=lambda name, timeout: FailedClient(),
    )

    with caplog.at_level("WARNING", logger="gemini"), pytest.raises(GeminiRequestError) as raised:
        model.invoke(["test"])

    assert secret not in str(raised.value)
    assert secret not in caplog.text


@pytest.mark.parametrize(
    ("category", "expected_message"),
    [
        ("temporary", "временно перегружен"),
        ("quota", "квоты"),
        ("authorization", "отклонил авторизацию"),
        ("configuration", "настройки доступа или запроса"),
    ],
)
def test_web_ui_explains_gemini_failure_category(
    category: str,
    expected_message: str,
) -> None:
    from web_app import _friendly_error

    assert expected_message in _friendly_error(
        GeminiRequestError("sanitized detail", category=category)
    )


def test_gemini_honors_retry_after_and_total_attempt_limit() -> None:
    delays: list[float] = []

    class Response:
        headers = {"Retry-After": "3"}

    class FailedClient:
        def invoke(self, messages: list[object], **kwargs: object) -> AIMessage:
            from google.genai.errors import ServerError

            error = ServerError(503, {"message": "temporary overload"})
            error.response = Response()
            raise error

    model = ResilientGeminiChat(
        api_key="test-only",
        preferred_model="gemini-3.8-flash",
        retry_budget=20,
        max_attempts=2,
        catalog_loader=lambda key, timeout: {
            "gemini-3.8-flash": {"generateContent"},
            "gemini-3.6-flash": {"generateContent"},
        },
        model_factory=lambda name, timeout: FailedClient(),
        sleep=delays.append,
    )

    try:
        model.invoke(["test"])
    except GeminiRequestError as error:
        assert error.category == "temporary"
        assert "Попыток: 2" in str(error)
    else:
        raise AssertionError("Исчерпание попыток должно возвращать понятную ошибку")

    assert delays == [3.0]


def test_gemini_bounds_each_request_by_remaining_budget() -> None:
    timeouts: list[float] = []

    class FailedClient:
        def invoke(self, messages: list[object], **kwargs: object) -> AIMessage:
            from google.genai.errors import ServerError

            raise ServerError(503, {"message": "temporary overload"})

    model = ResilientGeminiChat(
        api_key="test-only",
        preferred_model="gemini-3.8-flash",
        request_timeout=18,
        retry_budget=7,
        max_attempts=2,
        catalog_loader=lambda key, timeout: {"gemini-3.8-flash": {"generateContent"}},
        model_factory=lambda name, timeout: (timeouts.append(timeout) or FailedClient()),
        sleep=lambda delay: None,
        jitter=lambda low, high: 0,
    )

    try:
        model.invoke(["test"])
    except GeminiRequestError:
        pass
    else:
        raise AssertionError("Ожидалось исчерпание бюджета")

    assert len(timeouts) == 2
    assert all(0 < timeout <= 7 for timeout in timeouts)


def test_agent_logs_duration_and_preserves_provider_error(caplog) -> None:
    class FailedModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            raise TimeoutError("provider timed out")

    from agents import run_analyst

    with caplog.at_level("INFO", logger="agents"):
        try:
            run_analyst(FailedModel(), {"requirement": "Тест"})
        except TimeoutError as error:
            assert str(error) == "provider timed out"
        else:
            raise AssertionError("Ожидалась исходная ошибка провайдера")

    assert "Начат агент: аналитик требований" in caplog.text
    assert "Ошибка агента аналитик требований" in caplog.text
    assert "TimeoutError" in caplog.text