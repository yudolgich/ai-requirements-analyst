from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

BASE_CONTEXT = """Исходное требование:
{requirement}

Сохраненный контекст прошлых анализов (это не подтверждение заказчиком):
{prior_context}

Демонстрационные и найденные материалы (это источники рекомендаций, не факты о продукте):
{knowledge_context}

Материалы MCP:
{mcp_context}

Результаты предыдущих агентов:
{upstream_result}
"""
logger = logging.getLogger(__name__)


def _context(state: dict[str, Any]) -> str:
    chunks = state.get("rag_context", [])
    knowledge = "\n\n".join(
        f"Источник: {chunk['source']}\n{chunk['text']}" for chunk in chunks
    ) or "Подходящие фрагменты RAG не найдены."
    return BASE_CONTEXT.format(
        requirement=state["requirement"],
        prior_context=state.get("prior_context") or "Нет сохраненного контекста.",
        knowledge_context=knowledge,
        mcp_context=state.get("mcp_context") or "Контекст через MCP недоступен.",
        upstream_result=(
            state.get("upstream_result")
            or state.get("upstream_analysis")
            or "Это первый агент в цепочке."
        ),
    )


def _invoke(model: Any, role: str, instructions: str, state: dict[str, Any]) -> str:
    started_at = time.monotonic()
    logger.info("Начат агент: %s", role)
    try:
        response = model.invoke(
            [
                SystemMessage(content=f"Ты специализированный агент: {role}. {instructions}"),
                HumanMessage(content=_context(state)),
            ]
        )
    except Exception as error:
        elapsed = time.monotonic() - started_at
        detail = str(error)
        secret = os.getenv("GEMINI_API_KEY")
        if secret:
            detail = detail.replace(secret, "[скрыт]")
        logger.error(
            "Ошибка агента %s через %.1f сек. (%s): %s",
            role,
            elapsed,
            type(error).__name__,
            detail,
        )
        raise
    logger.info("Агент завершен: %s (%.1f сек.)", role, time.monotonic() - started_at)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def run_analyst(model: Any, state: dict[str, Any]) -> str:
    return _invoke(
        model,
        "аналитик требований",
        "Оцени ясность, полноту, однозначность и проверяемость. Раздели функциональные и нефункциональные аспекты. Перечисли пробелы, противоречия, подтвержденные факты и предположения. Не придумывай бизнес-правила.",
        state,
    )


def run_clarifier(model: Any, state: dict[str, Any]) -> str:
    return _invoke(
        model,
        "специалист по уточнению требований",
        "Используй исходное требование и анализ аналитика ниже. Составь конкретные вопросы заказчику, сгруппируй их по темам, отдельно перечисли подтвержденные факты и предположения. Не отвечай на вопросы за заказчика и не превращай предположения в факты.",
        {**state, "upstream_analysis": state.get("analysis", "")},
    )


def run_formalizer(model: Any, state: dict[str, Any]) -> str:
    enriched = dict(state)
    enriched["upstream_result"] = (
        f"АНАЛИЗ ТРЕБОВАНИЯ: {state.get('analysis', '')}"
        f"\n\nВОПРОСЫ И РАЗДЕЛЕНИЕ ФАКТОВ/ПРЕДПОЛОЖЕНИЙ: {state.get('clarification', '')}"
    )
    return _invoke(
        model,
        "специалист по формализации требований",
        "Сформулируй улучшенную версию, сохраняя исходный смысл. Создай пользовательскую историю (если уместно), проверяемые критерии Given/When/Then, позитивные, негативные и граничные сценарии. Не включай неподтвержденные ответы и правила как факты; неизвестное явно помечай.",
        enriched,
    )


def run_quality_reviewer(model: Any, state: dict[str, Any]) -> str:
    enriched = dict(state)
    enriched["upstream_result"] = (
        f"АНАЛИЗ: {state.get('analysis', '')}"
        f"\n\nВОПРОСЫ И ФАКТЫ/ПРЕДПОЛОЖЕНИЯ: {state.get('clarification', '')}"
        f"\n\nФОРМАЛИЗАЦИЯ: {state.get('formalization', '')}"
    )
    return _invoke(
        model,
        "контролер качества требований",
        "Проверь формализацию и предыдущие результаты на противоречия с исходными данными, неоднозначность, полноту и тестируемость. Назови конкретные проблемы и исправления. Не добавляй бизнес-правила как подтвержденные факты. Если проблем нет, явно скажи это.",
        enriched,
    )