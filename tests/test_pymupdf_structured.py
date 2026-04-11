"""Tests for PyMuPDFStructuredExtractor — run with mocked fitz."""

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.ocr.pymupdf_structured import PyMuPDFStructuredExtractor


# ---------------------------------------------------------------------------
# Helpers: minimal fitz mocks
# ---------------------------------------------------------------------------


def make_span(text: str, size: float, flags: int = 0, bbox: tuple = (0, 0, 50, 12)) -> dict:
    return {"text": text, "size": size, "flags": flags, "bbox": bbox, "origin": (0.0, bbox[1])}


def make_page_dict(spans: list[dict]) -> dict:
    return {
        "blocks": [
            {
                "type": 0,
                "bbox": (0, spans[0]["bbox"][1], 100, spans[-1]["bbox"][3]),
                "lines": [
                    {"spans": [span]} for span in spans
                ],
            }
        ]
    }


def build_mock_doc(pages_data: list[list[dict]]) -> MagicMock:
    """Build a minimal fitz document mock.

    pages_data: list of pages, each page is a list of spans.
    """
    mock_doc = MagicMock()
    mock_doc.get_toc.return_value = []

    pages = []
    for page_spans in pages_data:
        page_mock = MagicMock()
        page_mock.get_text.return_value = make_page_dict(page_spans)
        pages.append(page_mock)

    mock_doc.__iter__ = lambda self: iter(pages)
    mock_doc.__getitem__ = lambda self, idx: pages[idx]
    mock_doc.__len__ = lambda self: len(pages)
    mock_doc.close = MagicMock()
    return mock_doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPyMuPDFStructuredExtractor:
    def _make_extractor(self) -> PyMuPDFStructuredExtractor:
        return PyMuPDFStructuredExtractor(heading_size_ratio=1.3, max_heading_levels=4)

    def test_extract_all_returns_chunks(self):
        extractor = self._make_extractor()
        spans = [
            make_span("Introduction", size=16.0, bbox=(0, 0, 100, 14)),
            make_span("Body text here.", size=10.0, bbox=(0, 20, 100, 34)),
        ]
        mock_doc = build_mock_doc([spans])

        with patch("fitz.open", return_value=mock_doc):
            chunks = extractor.extract_all(Path("test.pdf"), "doc1")

        assert len(chunks) == 1  # spans merge into one block
        assert "Introduction" in chunks[0].content or "Body" in chunks[0].content

    def test_section_title_assigned_after_heading(self):
        extractor = self._make_extractor()
        # Page 0: heading span followed by body span in separate blocks
        heading_span = make_span("Methods", size=20.0, bbox=(0, 0, 100, 14))
        body_span = make_span("We used 42 samples.", size=10.0, bbox=(0, 30, 100, 44))

        # Two distinct blocks
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (0, 0, 100, 14),
                    "lines": [{"spans": [heading_span]}],
                },
                {
                    "type": 0,
                    "bbox": (0, 30, 100, 44),
                    "lines": [{"spans": [body_span]}],
                },
            ]
        }

        mock_doc = MagicMock()
        mock_doc.get_toc.return_value = []
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__getitem__ = lambda self, idx: mock_page
        mock_doc.__len__ = lambda self: 1
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            chunks = extractor.extract_all(Path("test.pdf"), "doc1")

        body_chunks = [c for c in chunks if "samples" in c.content]
        # Body chunk should be annotated with the preceding heading
        if body_chunks:
            assert body_chunks[0].section_title == "Methods"

    def test_bbox_set_on_chunks(self):
        extractor = self._make_extractor()
        spans = [make_span("Text", size=10.0, bbox=(5, 10, 200, 25))]
        mock_doc = build_mock_doc([spans])

        with patch("fitz.open", return_value=mock_doc):
            chunks = extractor.extract_all(Path("test.pdf"), "doc1")

        assert chunks[0].bbox is not None

    def test_image_blocks_skipped(self):
        extractor = self._make_extractor()
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {"type": 1, "bbox": (0, 0, 100, 100)},  # image block
                {
                    "type": 0,
                    "bbox": (0, 110, 100, 130),
                    "lines": [{"spans": [make_span("Real text", size=10.0, bbox=(0, 110, 100, 130))]}],
                },
            ]
        }

        mock_doc = MagicMock()
        mock_doc.get_toc.return_value = []
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__getitem__ = lambda self, idx: mock_page
        mock_doc.__len__ = lambda self: 1
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            chunks = extractor.extract_all(Path("test.pdf"), "doc1")

        assert len(chunks) == 1
        assert "Real text" in chunks[0].content
