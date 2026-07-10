"""FastAPI application exposing the RAG service over HTTP."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .models import Answer
from .service import RagService

_service: RagService | None = None


def get_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the service (loads models / connects to the store) on startup.
    get_service()
    yield


app = FastAPI(
    title="Agentic RAG Q&A",
    description="Ask questions about your documents using an agentic RAG pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "inline"


class IngestDirectoryRequest(BaseModel):
    directory: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    indexed_chunks: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "documents": get_service().document_count}


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(request: IngestTextRequest) -> IngestResponse:
    service = get_service()
    indexed = service.ingest_text(request.text, source=request.source)
    return IngestResponse(indexed_chunks=indexed, total_chunks=service.document_count)


@app.post("/ingest/directory", response_model=IngestResponse)
def ingest_directory(request: IngestDirectoryRequest) -> IngestResponse:
    service = get_service()
    try:
        indexed = service.ingest_directory(request.directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IngestResponse(indexed_chunks=indexed, total_chunks=service.document_count)


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest) -> Answer:
    service = get_service()
    if service.document_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Ingest documents before asking questions.",
        )
    return service.ask(request.question)
