"""The agentic RAG workflow, orchestrated with LangGraph.

The graph implements a small but genuine agent loop:

    START -> route -> (retrieve -> generate) | direct -> END

The ``route`` node decides whether a question needs document retrieval at all
(e.g. greetings and meta questions do not). When retrieval is needed, the
``retrieve`` node queries the vector store and ``generate`` produces an answer
grounded in the retrieved context, with inline citations.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import LLMClient
from .models import Answer, Citation, RetrievedChunk
from .pipeline import Retriever

SYSTEM_PROMPT = (
    "You are a precise assistant that answers strictly from the provided context. "
    "Cite the context blocks you use with inline markers like [1] or [2]. "
    "If the context does not contain the answer, say so plainly."
)

_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b",
    re.IGNORECASE,
)


class AgentState(TypedDict, total=False):
    question: str
    needs_retrieval: bool
    retrieved: list[RetrievedChunk]
    answer: Answer


def _build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context_lines = [
        f"[{i}] {rc.chunk.text}" for i, rc in enumerate(retrieved, start=1)
    ]
    context = "\n".join(context_lines) if context_lines else "(no context found)"
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def _citations(retrieved: list[RetrievedChunk]) -> list[Citation]:
    citations = []
    for i, rc in enumerate(retrieved, start=1):
        snippet = rc.chunk.text
        if len(snippet) > 160:
            snippet = snippet[:157].rstrip() + "..."
        citations.append(
            Citation(marker=f"[{i}]", source=rc.chunk.source, snippet=snippet)
        )
    return citations


def build_agent(retriever: Retriever, llm: LLMClient, *, top_k: int = 4):
    """Compile and return the LangGraph agent for the given retriever and LLM."""

    def route(state: AgentState) -> AgentState:
        question = state["question"]
        needs_retrieval = not bool(_SMALL_TALK.match(question))
        return {"needs_retrieval": needs_retrieval}

    def retrieve(state: AgentState) -> AgentState:
        retrieved = retriever.retrieve(state["question"], top_k=top_k)
        return {"retrieved": retrieved}

    def generate(state: AgentState) -> AgentState:
        question = state["question"]
        retrieved = state.get("retrieved", [])
        prompt = _build_prompt(question, retrieved)
        text = llm.generate(prompt, system=SYSTEM_PROMPT).strip()
        return {
            "answer": Answer(
                question=question,
                answer=text,
                citations=_citations(retrieved),
                used_retrieval=True,
            )
        }

    def direct(state: AgentState) -> AgentState:
        question = state["question"]
        text = llm.generate(
            f"Question: {question}\nAnswer:", system=SYSTEM_PROMPT
        ).strip()
        return {
            "answer": Answer(
                question=question,
                answer=text or "How can I help you with the indexed documents?",
                citations=[],
                used_retrieval=False,
            )
        }

    def _decide(state: AgentState) -> str:
        return "retrieve" if state.get("needs_retrieval", True) else "direct"

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("direct", direct)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _decide, {"retrieve": "retrieve", "direct": "direct"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("direct", END)

    return graph.compile()


class RagAgent:
    """Convenience wrapper around the compiled LangGraph agent."""

    def __init__(self, retriever: Retriever, llm: LLMClient, *, top_k: int = 4) -> None:
        self._graph = build_agent(retriever, llm, top_k=top_k)

    def ask(self, question: str) -> Answer:
        result = self._graph.invoke({"question": question})
        return result["answer"]
