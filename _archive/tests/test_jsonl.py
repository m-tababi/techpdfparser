"""Tests for JSONL read/write utilities."""
from __future__ import annotations

from src.core.models.elements import TextChunk
from src.utils.jsonl import read_jsonl, write_jsonl


def _make_chunk(i: int) -> TextChunk:
    return TextChunk(
        object_id=f"tc{i:04d}",
        doc_id="doc1",
        source_file="test.pdf",
        page_number=i,
        tool_name="mock",
        tool_version="0.0",
        content=f"chunk {i}",
        embedding=[float(i)] * 4,
    )


class TestJsonl:
    def test_write_returns_count(self, tmp_path):
        chunks = [_make_chunk(i) for i in range(3)]
        count = write_jsonl(tmp_path / "out.jsonl", chunks)
        assert count == 3

    def test_round_trip_preserves_content(self, tmp_path):
        chunks = [_make_chunk(i) for i in range(5)]
        path = tmp_path / "chunks.jsonl"
        write_jsonl(path, chunks)
        lines = list(read_jsonl(path))
        assert len(lines) == 5
        assert lines[2]["content"] == "chunk 2"

    def test_embedding_survives_serialization(self, tmp_path):
        chunk = _make_chunk(0)
        chunk.embedding = [0.1, 0.2, 0.3, 0.4]
        path = tmp_path / "e.jsonl"
        write_jsonl(path, [chunk])
        result = list(read_jsonl(path))
        assert result[0]["embedding"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.jsonl"
        write_jsonl(path, [_make_chunk(0)])
        assert path.exists()

    def test_empty_iterable_produces_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        count = write_jsonl(path, [])
        assert count == 0
        assert path.read_text() == ""


import pytest
