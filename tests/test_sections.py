"""Tests for section detection, assignment, and persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from src.core.models.document import BoundingBox
from src.core.models.elements import TextChunk
from src.utils.sections import (
    SectionMarker,
    assign_sections,
    detect_sections_from_fonts,
    load_sections,
    write_sections,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(
    page: int = 0,
    y0: float = 0.0,
    content: str = "text",
    section_title: str | None = None,
) -> TextChunk:
    chunk = TextChunk(
        object_id=f"c_{page}_{y0}",
        doc_id="doc1",
        source_file="test.pdf",
        page_number=page,
        tool_name="test",
        tool_version="1.0",
        content=content,
        bbox=BoundingBox(x0=0, y0=y0, x1=100, y1=y0 + 10),
    )
    chunk.section_title = section_title
    return chunk


def make_span(text: str, size: float, flags: int = 0, page: int = 0, y: float = 0.0) -> dict:
    return {
        "text": text,
        "size": size,
        "flags": flags,
        "page": page,
        "origin": (0.0, y),
    }


# ---------------------------------------------------------------------------
# detect_sections_from_fonts
# ---------------------------------------------------------------------------


class TestDetectSectionsFromFonts:
    def test_returns_empty_for_no_spans(self):
        assert detect_sections_from_fonts([]) == []

    def test_detects_large_font_as_heading(self):
        spans = [
            make_span("Introduction", size=16.0, page=0, y=10.0),
            make_span("body text here", size=10.0, page=0, y=30.0),
            make_span("more body", size=10.0, page=0, y=50.0),
        ]
        # median = 10, threshold = 10 * 1.3 = 13 → size 16 qualifies
        markers = detect_sections_from_fonts(spans, size_ratio=1.3)
        assert len(markers) == 1
        assert markers[0].title == "Introduction"
        assert markers[0].level == 1
        assert markers[0].page == 0

    def test_bold_flag_triggers_heading(self):
        spans = [
            make_span("Bold Heading", size=10.0, flags=16, page=0, y=5.0),
            make_span("normal text", size=10.0, flags=0, page=0, y=20.0),
        ]
        markers = detect_sections_from_fonts(spans, size_ratio=1.3)
        assert any(m.title == "Bold Heading" for m in markers)

    def test_two_levels_ordered_by_size(self):
        # Need enough body spans to keep median low (= 10) so both heading sizes qualify
        body = [make_span(f"body {i}", size=10.0, page=0, y=float(200 + i * 20)) for i in range(5)]
        spans = [
            make_span("Chapter 1", size=20.0, page=0, y=0.0),
            make_span("Section 1.1", size=15.0, page=0, y=50.0),
            *body,
        ]
        markers = detect_sections_from_fonts(spans, size_ratio=1.2)
        levels = {m.title: m.level for m in markers}
        assert levels["Chapter 1"] < levels["Section 1.1"]

    def test_path_accumulates_hierarchy(self):
        body = [make_span(f"body {i}", size=10.0, page=0, y=float(200 + i * 20)) for i in range(5)]
        spans = [
            make_span("Chapter 1", size=20.0, page=0, y=0.0),
            make_span("Section 1.1", size=15.0, page=0, y=50.0),
            *body,
        ]
        markers = detect_sections_from_fonts(spans, size_ratio=1.2)
        sec_marker = next(m for m in markers if m.title == "Section 1.1")
        assert "Chapter 1" in sec_marker.path
        assert "Section 1.1" in sec_marker.path


# ---------------------------------------------------------------------------
# assign_sections
# ---------------------------------------------------------------------------


class TestAssignSections:
    def test_no_markers_leaves_blocks_unchanged(self):
        blocks = [make_chunk(page=0, y0=10.0)]
        assign_sections(blocks, [])
        assert blocks[0].section_title is None

    def test_block_inherits_preceding_marker(self):
        markers = [SectionMarker(page=0, y0=0.0, level=1, title="Intro", path=["Intro"])]
        blocks = [make_chunk(page=0, y0=20.0)]
        assign_sections(blocks, markers)
        assert blocks[0].section_title == "Intro"
        assert blocks[0].section_path == ["Intro"]

    def test_block_before_first_marker_gets_none(self):
        markers = [SectionMarker(page=0, y0=50.0, level=1, title="Methods", path=["Methods"])]
        blocks = [make_chunk(page=0, y0=10.0)]
        assign_sections(blocks, markers)
        # The block at y0=10 comes before the marker at y0=50 → no assignment
        assert blocks[0].section_title is None

    def test_multiple_markers_correct_assignment(self):
        markers = [
            SectionMarker(page=0, y0=0.0, level=1, title="Intro", path=["Intro"]),
            SectionMarker(page=0, y0=100.0, level=1, title="Methods", path=["Methods"]),
        ]
        block_intro = make_chunk(page=0, y0=50.0)
        block_methods = make_chunk(page=0, y0=150.0)
        assign_sections([block_intro, block_methods], markers)
        assert block_intro.section_title == "Intro"
        assert block_methods.section_title == "Methods"

    def test_marker_on_next_page_not_inherited_by_prev_page(self):
        markers = [SectionMarker(page=1, y0=0.0, level=1, title="Chapter 2", path=["Chapter 2"])]
        block = make_chunk(page=0, y0=999.0)  # last block on page 0
        assign_sections([block], markers)
        assert block.section_title is None


# ---------------------------------------------------------------------------
# write_sections / load_sections roundtrip
# ---------------------------------------------------------------------------


class TestSectionsPersistence:
    def test_roundtrip(self):
        markers = [
            SectionMarker(page=0, y0=0.0, level=1, title="Intro", path=["Intro"]),
            SectionMarker(page=1, y0=30.0, level=2, title="1.1 Background", path=["Intro", "1.1 Background"]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sections.json"
            write_sections(path, markers)
            loaded = load_sections(path)

        assert len(loaded) == 2
        assert loaded[0].title == "Intro"
        assert loaded[1].page == 1
        assert loaded[1].path == ["Intro", "1.1 Background"]

    def test_writes_valid_json(self):
        markers = [SectionMarker(page=0, y0=5.0, level=1, title="Test", path=["Test"])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sections.json"
            write_sections(path, markers)
            data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert data[0]["title"] == "Test"
