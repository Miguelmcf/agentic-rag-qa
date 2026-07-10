"""Shared test fixtures.

Tests use the dependency-free ``HashingEmbedder``, the in-memory vector store,
and the offline ``EchoLLM`` so the whole suite runs fast and without network
access or model downloads.
"""

from __future__ import annotations

import pytest

from agentic_rag.embeddings import HashingEmbedder
from agentic_rag.llm import EchoLLM
from agentic_rag.pipeline import Retriever
from agentic_rag.vectorstore import InMemoryVectorStore

DOCS = {
    "solar.md": (
        "Jupiter is the largest planet in the Solar System. "
        "Saturn is famous for its bright ring system made of ice particles. "
        "Mars is called the Red Planet because of iron oxide on its surface."
    ),
    "rag.md": (
        "Retrieval-Augmented Generation grounds language models in external "
        "documents. A vector database stores embeddings for fast similarity "
        "search, which reduces hallucinations."
    ),
}


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=256)


@pytest.fixture
def retriever(embedder: HashingEmbedder) -> Retriever:
    from agentic_rag.ingestion import chunk_text

    store = InMemoryVectorStore()
    retriever = Retriever(embedder, store)
    for source, text in DOCS.items():
        retriever.index(chunk_text(text, source=source, chunk_size=200, chunk_overlap=40))
    return retriever


@pytest.fixture
def llm() -> EchoLLM:
    return EchoLLM()
