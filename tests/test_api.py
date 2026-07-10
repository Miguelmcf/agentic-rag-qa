from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Force the offline, dependency-free stack for API tests.
    monkeypatch.setenv("RAG_LLM_PROVIDER", "echo")
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("RAG_VECTOR_STORE", "memory")

    from agentic_rag import api, config

    config.get_settings.cache_clear()
    importlib.reload(api)
    api._service = None  # ensure a fresh service per test
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_zero_documents_initially(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["documents"] == 0


def test_ask_without_documents_returns_400(client):
    response = client.post("/ask", json={"question": "anything?"})
    assert response.status_code == 400


def test_ingest_then_ask_flow(client):
    ingest = client.post(
        "/ingest/text",
        json={
            "text": "Jupiter is the largest planet in the Solar System.",
            "source": "space.md",
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["indexed_chunks"] >= 1

    answer = client.post("/ask", json={"question": "Which planet is the largest?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["used_retrieval"] is True
    assert body["answer"]
