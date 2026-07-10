"""Document loading and chunking."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import Chunk

_SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}
_WHITESPACE_RE = re.compile(r"\s+")


def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode()).hexdigest()[:12]
    return f"{Path(source).stem}-{index}-{digest}"


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split ``text`` into overlapping character windows on word boundaries."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - chunk_overlap
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        window = cleaned[start:end]
        # Avoid cutting a word in half when we are not at the end of the text.
        if end < len(cleaned):
            last_space = window.rfind(" ")
            if last_space > step // 2:
                window = window[:last_space]
                end = start + last_space
        stripped = window.strip()
        if stripped:
            chunks.append(
                Chunk(
                    id=_chunk_id(source, index, stripped),
                    text=stripped,
                    source=source,
                    metadata={"chunk_index": str(index)},
                )
            )
            index += 1
        if end >= len(cleaned):
            break
        start = end - chunk_overlap
    return chunks


def load_documents(
    directory: str | Path,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Load and chunk every supported text file under ``directory``."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            chunks.extend(
                chunk_text(
                    text,
                    source=path.name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
    return chunks
