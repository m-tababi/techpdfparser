"""Test that Pydantic models serialize, deserialize, and validate correctly."""

from datetime import datetime

import pytest

from src.core.models.document import BoundingBox, DocumentMeta
from src.core.models.elements import (
    Figure,
    Formula,
    Table,
    TextChunk,
    VisualPage,
)
from src.core.models.results import BenchmarkEntry, BenchmarkReport, FusionResult, RetrievalResult


def make_base_fields(**overrides):
    return {
        "object_id": "abc123",
        "doc_id": "doc1",
        "source_file": "/data/test.pdf",
        "page_number": 0,
        "tool_name": "test_tool",
        "tool_version": "1.0",
        **overrides,
    }


class TestBoundingBox:
    def test_dimensions(self):
        bbox = BoundingBox(x0=10, y0=20, x1=110, y1=70)
        assert bbox.width == 100
        assert bbox.height == 50

    def test_roundtrip(self):
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=200)
        assert BoundingBox.model_validate(bbox.model_dump()) == bbox


class TestDocumentMeta:
    def test_defaults(self):
        meta = DocumentMeta(
            doc_id="d1", source_file="a.pdf", total_pages=5, file_size_bytes=1024
        )
        assert isinstance(meta.ingested_at, datetime)
        assert meta.extra == {}

    def test_roundtrip(self):
        meta = DocumentMeta(
            doc_id="d1", source_file="a.pdf", total_pages=5, file_size_bytes=1024
        )
        assert DocumentMeta.model_validate(meta.model_dump()) == meta


class TestVisualPage:
    def test_create(self):
        page = VisualPage(**make_base_fields(), image_path="/out/img.png")
        assert page.object_type == "visual_page"
        assert page.embedding is None

    def test_with_embedding(self):
        page = VisualPage(
            **make_base_fields(),
            image_path="/out/img.png",
            embedding=[[0.1, 0.2], [0.3, 0.4]],
        )
        assert len(page.embedding) == 2


class TestTextChunk:
    def test_create(self):
        chunk = TextChunk(**make_base_fields(), content="Hello world")
        assert chunk.object_type == "text_chunk"
        assert chunk.char_start is None

    def test_optional_fields(self):
        chunk = TextChunk(
            **make_base_fields(), content="text", char_start=0, char_end=10
        )
        assert chunk.char_end == 10


class TestTable:
    def test_defaults(self):
        table = Table(**make_base_fields(), content="| a | b |\n| 1 | 2 |")
        assert table.rows == []
        assert table.headers == []
        assert table.object_type == "table"


class TestFormula:
    def test_create(self):
        formula = Formula(
            **make_base_fields(), latex=r"E = mc^2", content="E = mc^2"
        )
        assert formula.object_type == "formula"
        assert formula.latex == r"E = mc^2"


class TestFigure:
    def test_create(self):
        fig = Figure(**make_base_fields(), image_path="/out/fig.png")
        assert fig.object_type == "figure"
        assert fig.description is None
        assert fig.caption is None


class TestRetrievalResult:
    def test_create(self):
        chunk = TextChunk(**make_base_fields(), content="text")
        result = RetrievalResult(element=chunk, score=0.95, collection="text_chunks")
        assert result.score == pytest.approx(0.95)
        assert result.element.object_type == "text_chunk"


class TestBenchmarkReport:
    def test_summary_averages(self):
        entries = [
            BenchmarkEntry(
                pipeline_name="visual",
                tool_name="colqwen25",
                tool_version="0.2",
                doc_id="d1",
                total_elements=10,
                elapsed_seconds=2.0,
            ),
            BenchmarkEntry(
                pipeline_name="visual",
                tool_name="colqwen25",
                tool_version="0.2",
                doc_id="d2",
                total_elements=8,
                elapsed_seconds=4.0,
            ),
        ]
        report = BenchmarkReport(entries=entries)
        summary = report.summary()
        assert summary["colqwen25"] == pytest.approx(3.0)
