"""Application configuration, loaded from environment variables or a `.env` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["echo", "openai", "gemini"]
EmbeddingBackend = Literal["hashing", "huggingface"]
VectorStoreBackend = Literal["memory", "chroma"]


class Settings(BaseSettings):
    """Runtime settings.

    Every value has a sensible default so the service runs offline out of the box.
    Override any of them through environment variables prefixed with ``RAG_``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: LLMProvider = "echo"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_temperature: float = 0.0

    # Embeddings. "huggingface" uses a real sentence-transformers model (default);
    # "hashing" is a dependency-free fallback used for offline/CI runs.
    embedding_backend: EmbeddingBackend = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    vector_store: VectorStoreBackend = "memory"
    chroma_path: str = ".chroma"

    # Retrieval / chunking
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()
