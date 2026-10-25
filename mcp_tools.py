from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(getattr(item, "text", item)) for item in value
        )
    return str(getattr(value, "content", value))


class FilesystemMCPBridge:
    def __init__(
        self,
        knowledge_dir: Path,
        *,
        client_factory: Any = MultiServerMCPClient,
    ) -> None:
        self.knowledge_dir = knowledge_dir.resolve()
        self.client_factory = client_factory

    def _server_config(self) -> dict[str, dict[str, Any]]:
        command = os.getenv("MCP_FILESYSTEM_COMMAND", "npx")
        configured_args = os.getenv("MCP_FILESYSTEM_ARGS")
        args = (
            json.loads(configured_args)
            if configured_args
            else ["-y", "@modelcontextprotocol/server-filesystem", str(self.knowledge_dir)]
        )
        return {
            "knowledge_files": {
                "command": command,
                "args": args,
                "transport": "stdio",
            }
        }

    async def _fetch(self) -> dict[str, Any]:
        client = self.client_factory(self._server_config())
        async with client.session("knowledge_files") as session:
            tools = await load_mcp_tools(session)
            tools_by_name = {tool.name: tool for tool in tools}
            if "list_directory" not in tools_by_name or "read_file" not in tools_by_name:
                raise RuntimeError("MCP-сервер не предоставляет list_directory и read_file")

            listing = await tools_by_name["list_directory"].ainvoke(
                {"path": str(self.knowledge_dir)}
            )
            documents = sorted(
                path for path in self.knowledge_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".md", ".txt", ".rst"}
            )[:5]
            read_results: list[str] = []
            for path in documents:
                content = await tools_by_name["read_file"].ainvoke({"path": str(path)})
                relative = path.relative_to(self.knowledge_dir).as_posix()
                read_results.append(f"Источник MCP: {relative}\n{_as_text(content)}")

            context = "\n\n".join(read_results)
            return {
                "context": context,
                "sources": [path.relative_to(self.knowledge_dir).as_posix() for path in documents],
                "tool_names": sorted(tools_by_name),
                "status": (
                    f"MCP подключен; обнаружено инструментов: {len(tools)}; "
                    f"прочитано документов: {len(read_results)}."
                    if context
                    else f"MCP подключен; обнаружено инструментов: {len(tools)}; документы не найдены. {_as_text(listing)}"
                ),
            }

    def fetch_knowledge_context(self) -> dict[str, Any]:
        try:
            return asyncio.run(self._fetch())
        except Exception as error:
            return {
                "context": "",
                "sources": [],
                "status": f"MCP недоступен ({type(error).__name__}); анализ продолжен без него.",
                "error": str(error),
            }