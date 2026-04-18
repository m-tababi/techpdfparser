"""Tests for the CPU-baseline pymupdf_text segmenter.

Creates a tiny synthetic PDF with PyMuPDF so the test stays hermetic.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from extraction.adapters.pymupdf_text_segmenter import PyMuPDFTextSegmenter
from extraction.models import ElementType
from extraction.registry import get_segmenter


@pytest.fixture
def two_page_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "synthetic.pdf"
    doc = fitz.open()
    page1 = doc.new_page(width=400, height=400)
    page1.insert_text((50, 80), "Hello page one.")
    page1.insert_text((50, 200), "Second paragraph on page one.")
    page2 = doc.new_page(width=400, height=400)
    page2.insert_text((50, 80), "Content on page two.")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_segmenter_registered() -> None:
    seg = get_segmenter("pymupdf_text")
    assert seg.tool_name == "pymupdf_text"


def test_segmenter_produces_text_regions(two_page_pdf: Path) -> None:
    seg = PyMuPDFTextSegmenter()
    regions = seg.segment(two_page_pdf)

    assert regions, "segmenter returned no regions"
    assert all(r.region_type == ElementType.TEXT for r in regions)
    # Each region carries text content directly.
    assert all(r.content is not None and r.content.text for r in regions)
    # Both pages represented.
    pages_covered = {r.page for r in regions}
    assert pages_covered == {0, 1}
    # Page 1 should have at least two blocks.
    assert sum(1 for r in regions if r.page == 0) >= 2
