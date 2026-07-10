"""Vector store backends.

* :class:`InMemoryVectorStore` - a small NumPy cosine-similarity index used by
  default and in tests.
* :class:`ChromaVectorStore` - a persistent vector database backed by Chroma
  (installed with the ``chroma`` extra).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .config import Settings
from .models import Chunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk embeddings and answers nearest-neighbour queries."""

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...


class InMemoryVectorStore:
    """Cosine-similarity search over embeddings held in memory.

    Embeddings are assumed to be L2-normalized, so a dot product equals cosine
    similarity.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings.")
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._matrix = (
            embeddings if self._matrix is None else np.vstack([self._matrix, embeddings])
        )

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if self._matrix is None or not self._chunks:
            return []
        query = query_embedding.reshape(-1)
        scores = self._matrix @ query
        top_k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top_indices
        ]

    def count(self) -> int:
        return len(self._chunks)


class ChromaVectorStore:
    """Persistent vector database backend using Chroma."""

    def __init__(self, path: str = ".chroma", collection: str = "documents") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The 'chroma' extra is required for ChromaVectorStore. "
                'Install it with: pip install -e ".[chroma]"'
            ) from exc
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, **c.metadata} for c in chunks],
        )

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding.reshape(-1).tolist()],
            n_results=top_k,
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        retrieved: list[RetrievedChunk] = []
        for cid, text, meta, distance in zip(ids, documents, metadatas, distances, strict=False):
            meta = dict(meta or {})
            source = str(meta.pop("source", "unknown"))
            retrieved.append(
                RetrievedChunk(
                    chunk=Chunk(id=cid, text=text, source=source, metadata=meta),
                    score=1.0 - float(distance),  # convert cosine distance to similarity
                )
            )
        return retrieved

    def count(self) -> int:
        return int(self._collection.count())


def build_vector_store(settings: Settings) -> VectorStore:
    """Instantiate the vector store selected in ``settings``."""
    if settings.vector_store == "chroma":
        return ChromaVectorStore(path=settings.chroma_path)
    return InMemoryVectorStore()
