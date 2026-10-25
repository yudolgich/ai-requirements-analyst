from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from app import DATABASE_PATH, KNOWLEDGE_DIR, ROOT
from gemini import GeminiRequestError
from graph import build_analysis_graph
from memory import AnalysisMemory
from mcp_tools import FilesystemMCPBridge
from rag import KnowledgeBase

load_dotenv(ROOT / ".env")

AGENT_TABS = (
    ("Анализ требований", "analysis"),
    ("Уточнение требований", "clarification"),
    ("Формализация и критерии приёмки", "formalization"),
    ("Контроль качества", "quality"),
)
GRAPH_STAGES = {
    "mcp_context": "Получение контекста MCP",
    "rag_retrieval": "Поиск по базе знаний RAG",
    "requirements_analyst": "Анализ требований",
    "clarification_specialist": "Уточнение требований",
    "requirements_formalizer": "Формализация и критерии приёмки",
    "quality_controller": "Контроль качества",
}


@st.cache_resource(show_spinner=False)
def _get_resources() -> tuple[
    AnalysisMemory,
    KnowledgeBase | None,
    FilesystemMCPBridge,
    str | None,
]:
    memory = AnalysisMemory(DATABASE_PATH)
    rag_warning = None
    try:
        knowledge = KnowledgeBase(KNOWLEDGE_DIR, ROOT / "data" / "chroma")
        knowledge.index_documents()
    except Exception as error:
        knowledge = None
        rag_warning = (
            f"Индекс RAG недоступен ({type(error).__name__}); "
            "анализ продолжится без поиска по базе знаний."
        )
    bridge = FilesystemMCPBridge(KNOWLEDGE_DIR)
    return memory, knowledge, bridge, rag_warning


def _friendly_error(error: Exception) -> str:
    if isinstance(error, GeminiRequestError):
        return {
            "quota": (
                "Gemini ограничил запрос из-за квоты или частоты обращений. "
                "Проверьте квоту и повторите анализ вручную позже."
            ),
            "authorization": (
                "Gemini отклонил авторизацию. Проверьте GEMINI_API_KEY "
                "и права доступа к API."
            ),
            "configuration": (
                "Gemini отклонил настройки доступа или запроса. Проверьте GEMINI_API_KEY, "
                "GEMINI_MODEL и доступность модели с методом generateContent."
            ),
            "temporary": (
                "Gemini временно перегружен или не ответил вовремя. "
                "Можно повторить анализ вручную."
            ),
            "connection": (
                "Не удалось подключиться к Gemini API. Проверьте сеть и повторите анализ."
            ),
        }.get(
            error.category,
            "Gemini не смог выполнить запрос. Проверьте настройки API и повторите анализ.",
        )
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return (
            "Не удалось подключиться к сервису анализа. Проверьте интернет-соединение "
            "и доступность Gemini API."
        )
    return (
        f"Анализ прерван ({type(error).__name__}). Проверьте настройки Gemini API "
        "и повторите попытку."
    )


def _claim_analysis_submission(
    state: Any,
    session_id: str,
    requirement: str,
    history: list[dict[str, Any]],
) -> bool:
    request_key = (session_id, requirement.strip())
    already_completed = any(
        item["status"] == "completed" and item["requirement"] == requirement.strip()
        for item in history
    )
    if already_completed or state.get("last_analysis_submission") == request_key:
        return False
    state["last_analysis_submission"] = request_key
    return True


def _display_result(
    requirement: str,
    result: dict[str, Any],
    *,
    complete: bool = True,
) -> None:
    if not complete:
        st.warning(
            "Это частичный результат: анализ не завершён. "
            "Успешные этапы сохранены; продолжите выполнение кнопкой повтора."
        )
    st.subheader("Исходное требование")
    st.markdown(requirement)

    tabs = st.tabs([title for title, _ in AGENT_TABS])
    for tab, (_, key) in zip(tabs, AGENT_TABS):
        with tab:
            answer = result.get(key)
            st.markdown(answer or "_Для этого результата нет сохранённого ответа._")

    st.subheader("Источники и подключения")
    rag_column, mcp_column = st.columns(2)
    with rag_column:
        with st.container(border=True):
            st.markdown("**RAG — база знаний**")
            st.write(result.get("rag_status") or "Статус RAG не сохранён.")
            sources = result.get("rag_sources") or []
            if sources:
                st.caption("Источники: " + ", ".join(sources))
    with mcp_column:
        with st.container(border=True):
            st.markdown("**MCP — файловый сервер**")
            st.write(result.get("mcp_status") or "Статус MCP не сохранён.")
            sources = result.get("mcp_sources") or []
            if sources:
                st.caption("Прочитано: " + ", ".join(sources))


def main() -> None:
    st.set_page_config(page_title="AI Requirements Analyst", layout="wide")
    st.markdown(
        """
        <style>
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stAlert"],
        .stApp [data-testid="stStatusWidget"] {
            color: #1f2933;
        }
        .stApp [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dce8e7;
            border-radius: 10px;
            padding: 12px 16px;
        }
        .stApp [data-testid="stTextArea"] textarea {
            background: #ffffff;
            color: #1f2933;
            caret-color: #1f2933;
        }
        .stApp [data-testid="stTextArea"] textarea::placeholder {
            color: #667085;
            opacity: 1;
        }
        .stApp [data-testid="stFormSubmitButton"] button {
            background: #2F6F6D;
            border-color: #2F6F6D;
            color: #ffffff;
        }
        .stApp [data-testid="stFormSubmitButton"] button:hover {
            background: #265c5a;
            border-color: #265c5a;
            color: #ffffff;
        }
        .stApp [data-testid="stTabs"] button[aria-selected="true"] {
            color: #2F6F6D;
            border-bottom-color: #2F6F6D;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    memory, knowledge, bridge, rag_warning = _get_resources()
    sessions = memory.list_sessions(limit=None)
    session_ids = [session["session_id"] for session in sessions]

    active_session = st.session_state.get("active_session_id")
    if active_session not in session_ids:
        active_session = session_ids[0] if session_ids else memory.create_session()
        st.session_state.active_session_id = active_session
        sessions = memory.list_sessions(limit=None)
        session_ids = [session["session_id"] for session in sessions]

    with st.sidebar:
        st.title("Рабочая область")
        if st.button("＋ Новая сессия", use_container_width=True):
            st.session_state.active_session_id = memory.create_session()
            st.rerun()

        active_session = st.selectbox(
            "Сессия",
            options=session_ids,
            index=session_ids.index(active_session),
            format_func=lambda session_id: next(
                (
                    f"{session['title']} · {session['analyses']} анализов"
                    for session in sessions
                    if session["session_id"] == session_id
                ),
                session_id[:8],
            ),
            key="active_session_id",
        )

        history = memory.get_history(active_session)
        if history:
            selected_index = st.selectbox(
                "История анализов",
                options=list(range(len(history))),
                index=len(history) - 1,
                format_func=lambda index: (
                    f"{history[index]['created_at'][:16].replace('T', ' ')} · "
                    f"{history[index]['requirement'][:48]}"
                    f"{' · незавершён' if history[index]['status'] != 'completed' else ''}"
                ),
                key=f"history_{active_session}",
            )
            incomplete_analysis = next(
                (item for item in reversed(history) if item["status"] != "completed"),
                None,
            )
        else:
            selected_index = None
            incomplete_analysis = None
            st.caption("В этой сессии пока нет сохранённых анализов.")

        st.divider()
        st.caption("История хранится в локальной базе SQLite.")

    st.title("AI Requirements Analyst")
    st.write("Уточните исходное требование и получите структурированный анализ.")

    if rag_warning:
        st.warning(rag_warning)
    if not os.getenv("GEMINI_API_KEY"):
        st.info(
            "Для запуска анализа добавьте GEMINI_API_KEY в .env. "
            "Сохранённую историю можно просматривать без API-ключа."
        )

    failed_analyses = st.session_state.setdefault("failed_analyses", {})
    failed_analysis = failed_analyses.get(active_session)
    if incomplete_analysis is not None:
        failed_analysis = {
            "analysis_id": incomplete_analysis["analysis_id"],
            "session_id": active_session,
            "requirement": incomplete_analysis["requirement"],
            "message": incomplete_analysis.get("error_message")
            or "Предыдущий запуск не был завершён. Сохранённые этапы можно продолжить из checkpoint.",
        }
        failed_analyses[active_session] = failed_analysis

    with st.form("requirement_form", clear_on_submit=False):
        requirement = st.text_area(
            "Исходное требование",
            placeholder="Например: Пользователь может восстановить пароль по электронной почте.",
            height=180,
        )
        submitted = st.form_submit_button(
            "Запустить анализ",
            type="primary",
            use_container_width=True,
            disabled=failed_analysis is not None,
        )

    retry_requested = False
    if failed_analysis:
        st.error(failed_analysis.get("message", "Анализ не был завершён."))
        st.info("Сначала возобновите незавершённый анализ или создайте новую сессию.")
        retry_requested = st.button(
            "Повторить анализ",
            key=f"retry-analysis-{active_session}",
            disabled=not bool(os.getenv("GEMINI_API_KEY")),
        )

    current_result: dict[str, Any] | None = None
    current_requirement: str | None = None
    partial_result: dict[str, Any] | None = None
    partial_requirement: str | None = None
    attempted_and_failed = False
    duplicate_submission = False
    if submitted:
        duplicate_submission = not _claim_analysis_submission(
            st.session_state,
            active_session,
            requirement,
            history,
        )
    if duplicate_submission:
        st.warning(
            "Это требование уже запускалось в текущей сессии интерфейса. "
            "Чтобы избежать повторного платного запроса, создайте новую сессию для нового запуска."
        )
    if (submitted and not duplicate_submission) or retry_requested:
        run_requirement = (
            str(failed_analysis["requirement"])
            if retry_requested and failed_analysis
            else requirement.strip()
        )
        if not run_requirement:
            st.warning("Введите требование перед запуском анализа.")
        elif not os.getenv("GEMINI_API_KEY"):
            st.error("Анализ не запущен: задайте GEMINI_API_KEY в файле .env.")
        else:
            analysis_id = (
                int(failed_analysis["analysis_id"])
                if retry_requested and failed_analysis
                else memory.save_analysis(
                    active_session,
                    run_requirement,
                    {},
                    status="in_progress",
                )
            )
            initial_state = {
                "requirement": run_requirement,
                "session_id": active_session,
                "prior_context": memory.build_prior_context(active_session),
            }
            updates: dict[str, Any] = {}
            graph_error: Exception | None = None
            with st.status("Подготовка анализа…", expanded=True) as status:
                progress = st.progress(0, text="Ожидание этапов графа")
                try:
                    with SqliteSaver.from_conn_string(str(DATABASE_PATH)) as checkpointer:
                        graph = build_analysis_graph(
                            rag_store=knowledge,
                            mcp_bridge=bridge,
                            checkpointer=checkpointer,
                        )
                        graph_config = {
                            "configurable": {"thread_id": active_session}
                        }
                        if retry_requested:
                            checkpoint = graph.get_state(graph_config).values
                            updates = dict(checkpoint)
                            stage_keys = {
                                "mcp_context": "mcp_status",
                                "rag_retrieval": "rag_status",
                                "requirements_analyst": "analysis",
                                "clarification_specialist": "clarification",
                                "requirements_formalizer": "formalization",
                                "quality_controller": "quality",
                            }
                            completed_stages = sum(
                                bool(checkpoint.get(key)) for key in stage_keys.values()
                            )
                            progress.progress(
                                completed_stages / len(GRAPH_STAGES),
                                text="Продолжение с сохранённого этапа",
                            )
                        else:
                            completed_stages = 0
                        try:
                            for event in graph.stream(
                                None if retry_requested else initial_state,
                                config=graph_config,
                                stream_mode="updates",
                            ):
                                for node_name, update in event.items():
                                    updates.update(update)
                                    if node_name in GRAPH_STAGES:
                                        completed_stages += 1
                                        progress.progress(
                                            completed_stages / len(GRAPH_STAGES),
                                            text=GRAPH_STAGES[node_name],
                                        )
                                        status.write(GRAPH_STAGES[node_name])
                                        memory.save_analysis(
                                            active_session,
                                            run_requirement,
                                            updates,
                                            status="in_progress",
                                            analysis_id=analysis_id,
                                        )
                        except Exception:
                            try:
                                updates = dict(graph.get_state(graph_config).values)
                            except Exception as state_error:
                                st.error(
                                    "Не удалось прочитать сохранённое состояние графа "
                                    f"({type(state_error).__name__}); сохранены ответы "
                                    "из уже полученных событий."
                                )
                            raise
                        updates = dict(
                            graph.get_state(graph_config).values
                        )
                        memory.save_analysis(
                            active_session,
                            run_requirement,
                            updates,
                            status="completed",
                            analysis_id=analysis_id,
                        )
                except Exception as error:
                    graph_error = error
                    status.update(label="Анализ не завершён", state="error")
                    failure_message = _friendly_error(error)
                    failed_analyses[active_session] = {
                        "analysis_id": analysis_id,
                        "session_id": active_session,
                        "requirement": run_requirement,
                        "message": failure_message,
                    }
                    st.error(failure_message)
                    try:
                        memory.save_analysis(
                            active_session,
                            run_requirement,
                            updates,
                            status="failed",
                            analysis_id=analysis_id,
                            error_message=failure_message,
                        )
                    except Exception as save_error:
                        st.error(
                            "Не удалось сохранить частичный результат в SQLite "
                            f"({type(save_error).__name__})."
                        )
                    if any(
                        updates.get(key)
                        for key in ("analysis", "clarification", "formalization", "quality")
                    ):
                        partial_result = updates
                        partial_requirement = run_requirement
                else:
                    failed_analyses.pop(active_session, None)
                    progress.progress(1.0, text="Анализ завершён")
                    status.update(
                        label="Анализ завершён",
                        state="complete",
                        expanded=False,
                    )
                    current_result = updates
                    current_requirement = run_requirement
            attempted_and_failed = graph_error is not None

    if current_result is not None and current_requirement is not None:
        _display_result(current_requirement, current_result)
    elif partial_result is not None and partial_requirement is not None:
        _display_result(partial_requirement, partial_result, complete=False)
    elif attempted_and_failed:
        st.warning(
            "Анализ не завершён, и завершённых ответов агентов пока нет. "
            "Состояние запуска сохранено для возможного продолжения."
        )
    elif selected_index is not None:
        saved_analysis = history[selected_index]
        st.caption(f"Сохранено: {saved_analysis['created_at']}")
        _display_result(
            saved_analysis["requirement"],
            saved_analysis["results"],
            complete=saved_analysis["status"] == "completed",
        )
    else:
        st.container(border=True).info(
            "Введите требование выше или выберите сохранённый анализ в боковой панели."
        )


if __name__ == "__main__":
    main()
