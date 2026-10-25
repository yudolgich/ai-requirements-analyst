from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AnalysisMemory:
    """Durable session history; the same SQLite file also stores LangGraph checkpoints."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    requirement TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS analyses_session_idx
                    ON analyses(session_id, analysis_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def create_session(self, title: str = "Новая сессия") -> str:
        session_id = str(uuid.uuid4())
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO sessions(session_id, title, created_at) VALUES (?, ?, ?)",
                (session_id, title, self._now()),
            )
        return session_id

    def list_sessions(self, limit: int | None = 20) -> list[dict[str, str]]:
        with closing(self._connect()) as connection, connection:
            query = """SELECT s.session_id, s.title, s.created_at, COUNT(a.analysis_id) AS analyses
                       FROM sessions s LEFT JOIN analyses a USING(session_id)
                       GROUP BY s.session_id ORDER BY s.created_at DESC"""
            parameters: tuple[int, ...] = ()
            if limit is not None:
                query += " LIMIT ?"
                parameters = (limit,)
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def save_analysis(
        self,
        session_id: str,
        requirement: str,
        state: dict[str, Any],
        *,
        status: str = "completed",
        analysis_id: int | None = None,
        error_message: str | None = None,
    ) -> int:
        if status not in {"in_progress", "failed", "completed"}:
            raise ValueError(f"Unsupported analysis status: {status}")
        saved_fields = (
            "analysis", "clarification", "formalization", "quality",
            "rag_sources", "rag_status", "mcp_sources", "mcp_status",
        )
        results = {key: state.get(key) for key in saved_fields}
        results["_status"] = status
        if error_message:
            results["_error_message"] = error_message
        with closing(self._connect()) as connection, connection:
            if analysis_id is None:
                cursor = connection.execute(
                    """INSERT INTO analyses(session_id, requirement, results_json, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        session_id,
                        requirement,
                        json.dumps(results, ensure_ascii=False),
                        self._now(),
                    ),
                )
                return int(cursor.lastrowid)

            cursor = connection.execute(
                """UPDATE analyses SET results_json = ?
                   WHERE analysis_id = ? AND session_id = ?""",
                (json.dumps(results, ensure_ascii=False), analysis_id, session_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Analysis record was not found in the requested session.")
            return analysis_id

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM analyses WHERE session_id = ? ORDER BY analysis_id",
                (session_id,),
            ).fetchall()
        history = []
        for row in rows:
            results = json.loads(row["results_json"])
            status = results.pop("_status", "completed")
            error_message = results.pop("_error_message", None)
            history.append(
                {
                    "analysis_id": row["analysis_id"],
                    "requirement": row["requirement"],
                    "created_at": row["created_at"],
                    "status": status,
                    "error_message": error_message,
                    "results": results,
                }
            )
        return history

    def build_prior_context(self, session_id: str, limit: int = 3) -> str:
        history = [
            item for item in self.get_history(session_id)
            if item["status"] == "completed"
        ][-limit:]
        if not history:
            return ""
        sections = []
        for item in history:
            results = item["results"]
            sections.append(
                f"Требование: {item['requirement']}\n"
                f"Формализация: {results.get('formalization') or 'не сохранена'}\n"
                f"Контроль качества: {results.get('quality') or 'не сохранен'}"
            )
        return "\n\n--- Предыдущий анализ ---\n\n".join(sections)