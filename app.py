
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_analysis_graph
from memory import AnalysisMemory
from mcp_tools import FilesystemMCPBridge
from rag import KnowledgeBase

ROOT = Path(__file__).resolve().parent
DATABASE_PATH = ROOT / "data" / "analysis.sqlite3"
KNOWLEDGE_DIR = ROOT / "knowledge_base"


def _safe_error_detail(error: Exception) -> str:
    detail = str(error)
    secret = os.getenv("GEMINI_API_KEY")
    return detail.replace(secret, "[скрыт]") if secret else detail


def _print_history(memory: AnalysisMemory, session_id: str) -> None:
    history = memory.get_history(session_id)
    if not history:
        print("В этой сессии пока нет анализов.")
        return
    for index, item in enumerate(history, start=1):
        results = item["results"]
        print(f"\n--- Анализ {index} ({item['created_at']}) ---")
        print(f"Требование: {item['requirement']}")
        print(f"Формализация:\n{results.get('formalization') or 'Нет данных'}")
        print(f"Контроль качества:\n{results.get('quality') or 'Нет данных'}")


def _choose_session(memory: AnalysisMemory) -> str | None:
    while True:
        sessions = memory.list_sessions()
        print("\nСессии: [1] новая  [2] продолжить  [3] история  [0] выход")
        choice = input("Выберите действие: ").strip()
        if choice == "0":
            return None
        if choice == "1":
            return memory.create_session()
        if choice not in {"2", "3"}:
            print("Введите 1, 2, 3 или 0.")
            continue
        if not sessions:
            print("Сохраненных сессий пока нет.")
            continue
        for index, session in enumerate(sessions, start=1):
            print(
                f"{index}. {session['title']} | анализов: {session['analyses']} "
                f"| id: {session['session_id'][:8]}"
            )
        selection = input("Номер сессии: ").strip()
        if not selection.isdigit() or not 1 <= int(selection) <= len(sessions):
            print("Неверный номер сессии.")
            continue
        session_id = sessions[int(selection) - 1]["session_id"]
        if choice == "3":
            _print_history(memory, session_id)
            continue
        return session_id


def _display_result(result: dict[str, Any]) -> None:
    sections = (
        ("Агент 1. Анализ требований", "analysis"),
        ("Агент 2. Вопросы и подтвержденные факты", "clarification"),
        ("Агент 3. Формализация и критерии приемки", "formalization"),
        ("Агент 4. Контроль качества", "quality"),
    )
    for title, key in sections:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}\n{result.get(key, 'Нет результата')}")

    print("\nИсточники RAG:")
    sources = result.get("rag_sources", [])
    print("\n".join(f"- {source}" for source in sources) if sources else "Подходящие материалы не найдены.")
    print(result.get("rag_status", ""))
    print("\nMCP:")
    print(result.get("mcp_status", "MCP не проверен."))
    mcp_sources = result.get("mcp_sources", [])
    if mcp_sources:
        print("Документы MCP: " + ", ".join(mcp_sources))


def _run_mcp_check(bridge: FilesystemMCPBridge) -> int:
    result = bridge.fetch_knowledge_context()
    print(result["status"])
    if result.get("sources"):
        print("Прочитанные документы: " + ", ".join(result["sources"]))
    if result.get("error"):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="AI Requirements Analyst")
    parser.add_argument("--check-mcp", action="store_true", help="Проверить реальный файловый MCP-сервер")
    parser.add_argument("--index-knowledge", action="store_true", help="Переиндексировать документы базы знаний")
    args = parser.parse_args(argv)

    knowledge = KnowledgeBase(KNOWLEDGE_DIR, ROOT / "data" / "chroma")
    try:
        indexed_chunks = knowledge.index_documents()
        print(f"База знаний: индексировано фрагментов: {indexed_chunks}.")
    except Exception as error:
        print(f"Индексирование недоступно ({type(error).__name__}); анализ продолжится без RAG.")
        knowledge = None  # type: ignore[assignment]

    bridge = FilesystemMCPBridge(KNOWLEDGE_DIR)
    if args.check_mcp:
        return _run_mcp_check(bridge)
    if args.index_knowledge:
        return 0 if knowledge is not None else 1

    memory = AnalysisMemory(DATABASE_PATH)
    with SqliteSaver.from_conn_string(str(DATABASE_PATH)) as checkpointer:
        graph = build_analysis_graph(
            rag_store=knowledge,
            mcp_bridge=bridge,
            checkpointer=checkpointer,
        )
        while True:
            session_id = _choose_session(memory)
            if session_id is None:
                return 0
            print(f"\nСессия {session_id[:8]}. Пустой ввод возвращает в меню.")
            while True:
                requirement = input("\nТребование: ").strip()
                if not requirement:
                    break
                if not os.getenv("GEMINI_API_KEY"):
                    print("Для анализа задайте GEMINI_API_KEY в .env. История остается доступной.")
                    continue
                try:
                    result = graph.invoke(
                        {
                            "requirement": requirement,
                            "session_id": session_id,
                            "prior_context": memory.build_prior_context(session_id),
                        },
                        config={"configurable": {"thread_id": session_id}},
                    )
                except Exception as error:
                    print(
                        f"Анализ не выполнен ({type(error).__name__}): "
                        f"{_safe_error_detail(error)}"
                    )
                    continue
                memory.save_analysis(session_id, requirement, result)
                _display_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
