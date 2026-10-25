from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeBase:
    """Persistent Chroma index using Chroma's local all-MiniLM embedding function."""

    SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}

    def __init__(self, documents_dir: Path, database_dir: Path) -> None:
        self.documents_dir = documents_dir
        self.database_dir = database_dir
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = Chroma(
            collection_name="requirements_knowledge",
            persist_directory=str(database_dir),
            embedding_function=None,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            add_start_index=True,
        )

    def index_documents(self) -> int:
        paths = sorted(
            path for path in self.documents_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_SUFFIXES
        )
        indexed = 0
        for path in paths:
            relative_path = path.relative_to(self.documents_dir).as_posix()
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                continue
            if not text:
                continue

            self.vector_store.delete(where={"source": relative_path})
            chunks = self.splitter.split_text(text)
            ids = [f"{relative_path}:{index}" for index in range(len(chunks))]
            metadata = [
                {"source": relative_path, "chunk": index}
                for index in range(len(chunks))
            ]
            self.vector_store.add_texts(chunks, metadatas=metadata, ids=ids)
            indexed += len(chunks)
        return indexed

    def search(self, query: str, limit: int = 4) -> list[dict[str, str]]:
        if not query.strip():
            return []
        results = self.vector_store.similarity_search(query, k=limit)
        return [
            {
                "source": str(document.metadata.get("source", "неизвестный источник")),
                "text": document.page_content,
            }
            for document in results
        ]