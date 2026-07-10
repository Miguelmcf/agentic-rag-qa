from __future__ import annotations

from agentic_rag.graph import RagAgent
from agentic_rag.llm import EchoLLM
from agentic_rag.pipeline import Retriever


def test_agent_answers_with_retrieval_and_citations(retriever: Retriever, llm: EchoLLM):
    agent = RagAgent(retriever, llm, top_k=2)
    answer = agent.ask("Which planet is the largest?")
    assert answer.used_retrieval is True
    assert answer.citations, "expected at least one citation"
    assert "planet" in answer.answer.lower() or "jupiter" in answer.answer.lower()


def test_agent_routes_small_talk_without_retrieval(retriever: Retriever, llm: EchoLLM):
    agent = RagAgent(retriever, llm)
    answer = agent.ask("Hello there!")
    assert answer.used_retrieval is False
    assert answer.citations == []


def test_agent_citation_markers_are_sequential(retriever: Retriever, llm: EchoLLM):
    agent = RagAgent(retriever, llm, top_k=2)
    answer = agent.ask("What does a vector database do?")
    markers = [c.marker for c in answer.citations]
    assert markers == [f"[{i}]" for i in range(1, len(markers) + 1)]
