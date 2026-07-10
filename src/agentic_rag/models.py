"""Core data models shared across the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable slice of a source document."""

    id: str
    text: str
    source: str
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever together with its similarity score."""

    chunk: Chunk
    score: float


class Citation(BaseModel):
    """A reference back to the source material used in an answer."""

    marker: str
    source: str
    snippet: str


class Answer(BaseModel):
    """The final grounded answer produced by the agent."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_retrieval: bool = True
