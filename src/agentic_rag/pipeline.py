"""Retrieval pipeline: ties an embedder to a vector store."""

from __future__ import annotations

from .embeddings import Embedder
from .models import Chunk, RetrievedChunk
from .vectorstore import VectorStore


class Retriever:
    """Indexes chunks and retrieves the most relevant ones for a query."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def index(self, chunks: list[Chunk]) -> int:
        """Embed and store ``chunks``. Returns the number indexed."""
        if not chunks:
            return 0
        embeddings = self._embedder.embed([c.text for c in chunks])
        self._store.add(chunks, embeddings)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        if not query.strip():
            return []
        query_embedding = self._embedder.embed([query])[0]
        return self._store.search(query_embedding, top_k)

    @property
    def size(self) -> int:
        return self._store.count()
