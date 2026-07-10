"""High-level service that wires the whole pipeline together from settings."""

from __future__ import annotations

from pathlib import Path

from .config import Settings, get_settings
from .embeddings import build_embedder
from .graph import RagAgent
from .ingestion import chunk_text, load_documents
from .llm import build_llm
from .models import Answer
from .pipeline import Retriever
from .vectorstore import build_vector_store


class RagService:
    """Orchestrates ingestion and question answering."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        embedder = build_embedder(self.settings)
        store = build_vector_store(self.settings)
        self.retriever = Retriever(embedder, store)
        self.agent = RagAgent(self.retriever, build_llm(self.settings), top_k=self.settings.top_k)

    def ingest_directory(self, directory: str | Path) -> int:
        chunks = load_documents(
            directory,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        return self.retriever.index(chunks)

    def ingest_text(self, text: str, source: str = "inline") -> int:
        chunks = chunk_text(
            text,
            source=source,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        return self.retriever.index(chunks)

    def ask(self, question: str) -> Answer:
        return self.agent.ask(question)

    @property
    def document_count(self) -> int:
        return self.retriever.size
