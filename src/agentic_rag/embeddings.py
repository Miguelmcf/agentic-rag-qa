"""Text embedding backends.

Two implementations are provided:

* :class:`HashingEmbedder` - a dependency-free, deterministic embedder used for
  offline demos and CI. It is not semantically strong but requires no downloads.
* :class:`HuggingFaceEmbedder` - real semantic embeddings via
  ``sentence-transformers`` (installed with the ``huggingface`` extra).

Both conform to the :class:`Embedder` protocol so the rest of the code never
depends on a concrete backend.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

from .config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_truststore_injected = False


def _use_os_truststore() -> None:
    """Route TLS verification through the OS trust store when available.

    This lets model downloads succeed on networks with a TLS-inspecting proxy
    (common in corporate environments) whose root CA lives in the system store
    rather than in ``certifi``. It is a no-op if ``truststore`` isn't installed.
    """
    global _truststore_injected
    if _truststore_injected:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - best-effort, never fatal
        pass
    _truststore_injected = True


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalized vectors."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedder (no external models)."""

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t) for t in texts])
        return _normalize(matrix)


class HuggingFaceEmbedder:
    """Semantic embeddings backed by ``sentence-transformers``."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None  # loaded lazily to keep import time and CI light

    def _load(self):
        if self._model is None:
            _use_os_truststore()
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - core dependency
                raise ImportError(
                    "sentence-transformers is required for HuggingFaceEmbedder. "
                    "Reinstall the project dependencies: pip install -e ."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._load().encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(settings: Settings) -> Embedder:
    """Instantiate the embedder selected in ``settings``."""
    if settings.embedding_backend == "huggingface":
        return HuggingFaceEmbedder(settings.embedding_model)
    return HashingEmbedder()
