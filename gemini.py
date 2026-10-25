from __future__ import annotations

import logging
import os
import random
import re
import socket
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import httpx
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
TRANSIENT_HTTP_CODES = {408, 425, 500, 502, 503, 504}


def _safe_error_text(error: Exception, api_key: str) -> str:
    detail = str(error)
    return detail.replace(api_key, "[скрыт]") if api_key else detail


class GeminiRequestError(RuntimeError):
    """The bounded request could not be completed by an available Gemini model."""

    def __init__(self, message: str, *, category: str = "unknown") -> None:
        super().__init__(message)
        self.category = category


def _normalize_model_name(name: str) -> str:
    return name.removeprefix("models/")


def _supports_generate_content(methods: set[str]) -> bool:
    return "generateContent" in methods


def _version_key(name: str) -> tuple[int, ...]:
    return tuple(int(value) for value in re.findall(r"\d+", name))


def select_catalog_model(
    available: dict[str, set[str]],
    preferred_model: str | None = None,
) -> str:
    verified_models = {
        _normalize_model_name(name)
        for name, methods in available.items()
        if _supports_generate_content(set(methods))
    }
    preferred = _normalize_model_name(preferred_model or "")
    if preferred and preferred in verified_models:
        return preferred
    if DEFAULT_GEMINI_MODEL in verified_models:
        return DEFAULT_GEMINI_MODEL
    if not verified_models:
        detail = "В каталоге API-ключа нет доступных моделей с generateContent."
    else:
        detail = (
            f"Безопасная модель по умолчанию {DEFAULT_GEMINI_MODEL} отсутствует в каталоге. "
            "Укажите GEMINI_MODEL явно, выбрав модель из проверенного списка."
        )
    raise GeminiRequestError(detail, category="configuration")


def load_text_generation_models(api_key: str, timeout_seconds: float) -> dict[str, set[str]]:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
    )
    available: dict[str, set[str]] = {}
    for model in client.models.list(config={"page_size": 1000}):
        name = _normalize_model_name(model.name or "").strip()
        methods = set(model.supported_actions or [])
        if name and _supports_generate_content(methods):
            available[name] = methods
    return available


def probe_text_generation_model(
    api_key: str,
    model_name: str,
    available_models: dict[str, set[str]],
    timeout_seconds: float = 8.0,
    *,
    model_factory: Callable[[str, str, float], Any] | None = None,
) -> Any:
    """Send exactly one short request to a catalog-verified model."""
    normalized_name = _normalize_model_name(model_name)
    if normalized_name not in {
        _normalize_model_name(name)
        for name, methods in available_models.items()
        if _supports_generate_content(set(methods))
    }:
        raise GeminiRequestError(
            f"Модель {normalized_name} не подтверждена каталогом generateContent.",
            category="configuration",
        )
    if timeout_seconds < 10:
        raise ValueError("The Gemini SDK requires a probe timeout of at least 10 seconds.")

    factory = model_factory or (
        lambda name, key, timeout: ChatGoogleGenerativeAI(
            model=name,
            api_key=key,
            temperature=0,
            request_timeout=timeout,
            retries=0,
        )
    )
    return factory(normalized_name, api_key, timeout_seconds).invoke(
        [HumanMessage(content="Ответь одним словом: OK.")]
    )


def _error_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    for value in (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(response, "status_code", None),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    match = re.match(r"\s*(400|401|403|404|408|425|429|500|502|503|504)\b", str(error))
    return int(match.group(1)) if match else None


def _error_category(error: Exception) -> str:
    code = _error_code(error)
    detail = str(error).lower()
    if any(
        marker in detail
        for marker in (
            "api_key_invalid",
            "invalid api key",
            "api key not valid",
            "unauthenticated",
            "authentication failed",
        )
    ):
        return "authorization"
    if code == 429:
        return "quota"
    if "resource_exhausted" in detail or "quota" in detail or "rate limit" in detail:
        return "quota"
    if code in {401, 403}:
        return "authorization"
    if code in {400, 404} or any(
        marker in detail for marker in ("invalid_argument", "not_found", "not found")
    ):
        return "configuration"
    if code in TRANSIENT_HTTP_CODES or any(
        marker in detail
        for marker in ("unavailable", "deadline_exceeded", "deadline exceeded", "gateway timeout")
    ):
        return "temporary"
    if isinstance(error, (TimeoutError, ConnectionError, socket.timeout, httpx.TransportError)):
        return "connection"
    return "unknown"


def _category_description(category: str) -> str:
    return {
        "quota": "Gemini ограничил запрос из-за квоты или частоты обращений.",
        "authorization": "Gemini отклонил авторизацию. Проверьте API-ключ и права доступа.",
        "configuration": "Gemini отклонил конфигурацию запроса или выбранную модель.",
        "temporary": "Gemini временно перегружен или не ответил вовремя.",
        "connection": "Не удалось подключиться к Gemini API.",
        "unknown": "Gemini не смог выполнить запрос.",
    }.get(category, "Gemini не смог выполнить запрос.")


def _request_error(
    error: Exception,
    api_key: str,
    *,
    attempts: int,
    elapsed: float,
) -> GeminiRequestError:
    category = _error_category(error)
    return GeminiRequestError(
        f"{_category_description(category)} "
        f"Попыток: {attempts}; время: {elapsed:.1f} сек. "
        f"Последняя ошибка ({type(error).__name__}): "
        f"{_safe_error_text(error, api_key)}",
        category=category,
    )


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        details = getattr(error, "details", None)
        if isinstance(details, dict):
            value = details.get("retryDelay") or details.get("retry_after")
    if value is None:
        return None
    try:
        return max(0.0, float(str(value).removesuffix("s")))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _is_transient(error: Exception) -> bool:
    return _error_category(error) in {"temporary", "quota", "connection"}


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content)


class ResilientGeminiChat:
    """Catalog-validated Gemini models with bounded per-call transient retries."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        preferred_model: str | None = None,
        request_timeout: float | None = None,
        retry_budget: float | None = None,
        max_attempts: int | None = None,
        catalog_loader: Callable[[str, float], dict[str, set[str]]] | None = None,
        model_factory: Callable[[str, float], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.preferred_model = _normalize_model_name(
            preferred_model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        )
        self.request_timeout = (
            request_timeout
            if request_timeout is not None
            else float(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "18"))
        )
        self.retry_budget = (
            retry_budget
            if retry_budget is not None
            else float(os.getenv("GEMINI_RETRY_BUDGET_SECONDS", "60"))
        )
        self.max_attempts = (
            max_attempts
            if max_attempts is not None
            else int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
        )
        if self.request_timeout <= 0 or self.retry_budget <= 0 or self.max_attempts <= 0:
            raise ValueError("Gemini request timeout, retry budget, and attempts must be positive.")
        self.catalog_loader = catalog_loader or load_text_generation_models
        self.model_factory = model_factory or self._create_langchain_model
        self.sleep = sleep
        self.jitter = jitter
        self._models: dict[str, set[str]] | None = None
        self._clients: dict[tuple[str, float], Any] = {}
        self._working_model: str | None = None

    @property
    def model_name(self) -> str:
        return self._working_model or self.preferred_model

    def _create_langchain_model(self, model_name: str, timeout: float) -> Any:
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.api_key,
            temperature=0.2,
            request_timeout=timeout,
            retries=0,
        )

    def _model_catalog(self, deadline: float | None = None) -> dict[str, set[str]]:
        if not self.api_key:
            raise GeminiRequestError(
                "GEMINI_API_KEY не задан.",
                category="configuration",
            )
        if self._models is None:
            started_at = time.monotonic()
            request_deadline = (
                deadline if deadline is not None else started_at + self.retry_budget
            )
            last_error: Exception | None = None
            catalog_attempts = 0
            for attempt in range(1, self.max_attempts + 1):
                remaining = request_deadline - time.monotonic()
                if remaining <= 0:
                    break
                catalog_attempts = attempt
                try:
                    catalog = self.catalog_loader(
                        self.api_key,
                        min(10.0, self.request_timeout, remaining),
                    )
                except Exception as error:
                    last_error = error
                    if not _is_transient(error) or attempt >= self.max_attempts:
                        request_error = _request_error(
                            error,
                            self.api_key,
                            attempts=attempt,
                            elapsed=time.monotonic() - started_at,
                        )
                        raise GeminiRequestError(
                            f"Не удалось получить каталог моделей Gemini. {request_error}",
                            category=request_error.category,
                        ) from error

                    delay = self._delay(error, attempt)
                    remaining = request_deadline - time.monotonic()
                    if delay >= remaining:
                        raise GeminiRequestError(
                            f"Повтор получения каталога отложен на {delay:.1f} сек., "
                            f"это превышает оставшийся лимит {max(0.0, remaining):.1f} сек. "
                            f"Последняя ошибка ({type(error).__name__}): "
                            f"{_safe_error_text(error, self.api_key)}",
                            category=_error_category(error),
                        ) from error
                    logger.warning(
                        "Сбой каталога Gemini (%s), попытка %d/%d (%s): %s",
                        _error_category(error),
                        attempt,
                        self.max_attempts,
                        type(error).__name__,
                        _safe_error_text(error, self.api_key),
                    )
                    self.sleep(delay)
                    continue

                self._models = {
                    _normalize_model_name(name): set(methods)
                    for name, methods in catalog.items()
                    if _supports_generate_content(set(methods))
                }
                break

            if self._models is None and last_error is not None:
                request_error = _request_error(
                    last_error,
                    self.api_key,
                    attempts=catalog_attempts,
                    elapsed=time.monotonic() - started_at,
                )
                raise GeminiRequestError(
                    f"Не удалось получить каталог моделей Gemini. {request_error}",
                    category=request_error.category,
                ) from last_error
        if not self._models:
            raise GeminiRequestError(
                "В каталоге API-ключа нет доступных моделей с generateContent.",
                category="configuration",
            )
        return self._models

    def available_models(self) -> list[str]:
        return sorted(self._model_catalog(), key=_version_key, reverse=True)

    def _selected_model(self, deadline: float | None = None) -> str:
        available = self._model_catalog(deadline)
        selected = select_catalog_model(available, self.preferred_model)
        if self.preferred_model and selected != self.preferred_model:
            logger.warning(
                "Настроенная модель %s отсутствует в каталоге; выбрана проверенная модель %s.",
                self.preferred_model,
                selected,
            )
        return selected

    def _client(self, name: str, timeout: float) -> Any:
        client_key = (name, timeout)
        if client_key not in self._clients:
            self._clients[client_key] = self.model_factory(name, timeout)
        return self._clients[client_key]

    def _delay(self, error: Exception, retry_number: int) -> float:
        ceiling = min(16.0, 2.0 * (2 ** (retry_number - 1)))
        backoff = self.jitter(ceiling / 2, ceiling)
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            return max(retry_after, backoff)
        return backoff

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        started_at = time.monotonic()
        deadline = started_at + self.retry_budget
        model_name = self._selected_model(deadline)
        last_error: Exception | None = None
        attempts = 0
        for attempt in range(self.max_attempts):
            elapsed = time.monotonic() - started_at
            remaining = self.retry_budget - elapsed
            if remaining <= 0:
                break
            request_timeout = min(self.request_timeout, remaining)
            attempts += 1
            try:
                response = self._client(model_name, request_timeout).invoke(
                    messages,
                    **kwargs,
                )
                self._working_model = model_name
                logger.info(
                    "Gemini вернул ответ на %s (попытка %d, %.1f сек.)",
                    model_name,
                    attempts,
                    time.monotonic() - started_at,
                )
                return response
            except Exception as error:
                last_error = error
                if not _is_transient(error):
                    raise _request_error(
                        error,
                        self.api_key,
                        attempts=attempts,
                        elapsed=time.monotonic() - started_at,
                    ) from error
                logger.warning(
                    "Сбой Gemini (%s) для %s, попытка %d/%d (%s): %s",
                    _error_category(error),
                    model_name,
                    attempts,
                    self.max_attempts,
                    type(error).__name__,
                    _safe_error_text(error, self.api_key),
                )
                if attempts >= self.max_attempts:
                    break
                delay = self._delay(error, attempts)
                remaining = self.retry_budget - (time.monotonic() - started_at)
                if delay >= remaining:
                    raise GeminiRequestError(
                        f"Повтор отложен провайдером на {delay:.1f} сек., "
                        f"это превышает оставшийся лимит {max(0.0, remaining):.1f} сек. "
                        f"Последняя ошибка: {type(error).__name__}: "
                        f"{_safe_error_text(error, self.api_key)}",
                        category=_error_category(error),
                    ) from error
                logger.info(
                    "Повтор %d/%d через %.1f сек. на той же модели %s.",
                    attempts + 1,
                    self.max_attempts,
                    delay,
                    model_name,
                )
                self.sleep(delay)

        if last_error is not None:
            raise _request_error(
                last_error,
                self.api_key,
                attempts=attempts,
                elapsed=time.monotonic() - started_at,
            ) from last_error
        raise GeminiRequestError(
            f"Лимит времени запроса к Gemini исчерпан "
            f"({time.monotonic() - started_at:.1f} сек.; попыток: {attempts}).",
            category="temporary",
        )


def test_prompt() -> HumanMessage:
    return HumanMessage(content="Ответь одним словом: OK.")