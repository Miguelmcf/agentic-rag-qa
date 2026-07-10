from __future__ import annotations

import pytest

from agentic_rag.ingestion import chunk_text, load_documents


def test_chunk_text_produces_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, source="doc.txt", chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(chunk.source == "doc.txt" for chunk in chunks)
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_chunk_text_empty_returns_nothing():
    assert chunk_text("   ", source="doc.txt") == []


def test_chunk_ids_are_unique():
    text = " ".join(f"token{i}" for i in range(300))
    chunks = chunk_text(text, source="doc.txt", chunk_size=150, chunk_overlap=30)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("hello world", source="d", chunk_size=100, chunk_overlap=100)


def test_load_documents_reads_sample_dir(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nSome content here.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("More content in a text file.", encoding="utf-8")
    (tmp_path / "ignore.pdf").write_text("not indexed", encoding="utf-8")

    chunks = load_documents(tmp_path, chunk_size=100, chunk_overlap=20)
    sources = {c.source for c in chunks}
    assert sources == {"a.md", "b.txt"}


def test_load_documents_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")
