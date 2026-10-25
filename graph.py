
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import (
    run_analyst,
    run_clarifier,
    run_formalizer,
    run_quality_reviewer,
)
from gemini import ResilientGeminiChat


class AnalysisState(TypedDict, total=False):
    requirement: str
    session_id: str
    prior_context: str
    rag_context: list[dict[str, str]]
    rag_sources: list[str]
    rag_status: str
    mcp_context: str
    mcp_sources: list[str]
    mcp_status: str
    analysis: str
    clarification: str
    formalization: str
    quality: str


def create_chat_model() -> Any:
    return ResilientGeminiChat()


def build_analysis_graph(
    *,
    model: Any | None = None,
    rag_store: Any | None = None,
    mcp_bridge: Any | None = None,
    checkpointer: Any | None = None,
) -> Any:
    llm = model or create_chat_model()

    def load_mcp_context(state: AnalysisState) -> dict[str, Any]:
        if mcp_bridge is None:
            return {"mcp_context": "", "mcp_sources": [], "mcp_status": "MCP не настроен."}
        result = mcp_bridge.fetch_knowledge_context()
        return {
            "mcp_context": result.get("context", ""),
            "mcp_sources": result.get("sources", []),
            "mcp_status": result.get("status", "Нет данных от MCP."),
        }

    def retrieve_knowledge(state: AnalysisState) -> dict[str, Any]:
        if rag_store is None:
            return {"rag_context": [], "rag_sources": [], "rag_status": "RAG не настроен."}
        try:
            chunks = rag_store.search(state["requirement"])
            return {
                "rag_context": chunks,
                "rag_sources": list(dict.fromkeys(chunk["source"] for chunk in chunks)),
                "rag_status": f"Найдено фрагментов: {len(chunks)}" if chunks else "Подходящие материалы не найдены.",
            }
        except Exception as error:
            return {
                "rag_context": [],
                "rag_sources": [],
                "rag_status": f"Поиск RAG недоступен ({type(error).__name__}). Анализ продолжен без него.",
            }

    def analyst(state: AnalysisState) -> dict[str, str]:
        return {"analysis": run_analyst(llm, state)}

    def clarifier(state: AnalysisState) -> dict[str, str]:
        return {"clarification": run_clarifier(llm, state)}

    def formalizer(state: AnalysisState) -> dict[str, str]:
        return {"formalization": run_formalizer(llm, state)}

    def quality_reviewer(state: AnalysisState) -> dict[str, str]:
        return {"quality": run_quality_reviewer(llm, state)}

    workflow = StateGraph(AnalysisState)
    workflow.add_node("mcp_context", load_mcp_context)
    workflow.add_node("rag_retrieval", retrieve_knowledge)
    workflow.add_node("requirements_analyst", analyst)
    workflow.add_node("clarification_specialist", clarifier)
    workflow.add_node("requirements_formalizer", formalizer)
    workflow.add_node("quality_controller", quality_reviewer)
    workflow.add_edge(START, "mcp_context")
    workflow.add_edge("mcp_context", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "requirements_analyst")
    workflow.add_edge("requirements_analyst", "clarification_specialist")
    workflow.add_edge("clarification_specialist", "requirements_formalizer")
    workflow.add_edge("requirements_formalizer", "quality_controller")
    workflow.add_edge("quality_controller", END)
    return workflow.compile(checkpointer=checkpointer)


def analyze_requirement(requirement: str) -> str:
    """Compatibility helper for callers that only need the analyst's result."""
    result = build_analysis_graph().invoke({"requirement": requirement})
    return result["analysis"]
