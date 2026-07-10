from __future__ import annotations

import numpy as np
import pytest

from agentic_rag.embeddings import HashingEmbedder
from agentic_rag.models import Chunk
from agentic_rag.pipeline import Retriever
from agentic_rag.vectorstore import InMemoryVectorStore


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dimension=128)
    a = embedder.embed(["hello world"])
    b = embedder.embed(["hello world"])
    assert np.allclose(a, b)
    assert a.shape == (1, 128)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_in_memory_store_add_and_count():
    store = InMemoryVectorStore()
    embedder = HashingEmbedder()
    chunks = [Chunk(id="1", text="alpha beta", source="s")]
    store.add(chunks, embedder.embed(["alpha beta"]))
    assert store.count() == 1


def test_in_memory_store_mismatched_lengths_raise():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.add([Chunk(id="1", text="x", source="s")], np.zeros((2, 4), dtype=np.float32))


def test_retriever_returns_relevant_chunk():
    embedder = HashingEmbedder()
    retriever = Retriever(embedder, InMemoryVectorStore())
    retriever.index(
        [
            Chunk(id="1", text="Jupiter is the largest planet", source="space.md"),
            Chunk(id="2", text="Python is a programming language", source="code.md"),
        ]
    )
    results = retriever.retrieve("Which planet is the largest?", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.source == "space.md"


def test_retriever_empty_query_returns_empty():
    retriever = Retriever(HashingEmbedder(), InMemoryVectorStore())
    assert retriever.retrieve("   ") == []
