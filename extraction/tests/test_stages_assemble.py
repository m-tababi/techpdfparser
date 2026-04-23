"""Unit tests for run_assemble."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from extraction.config import ExtractionConfig
from extraction.models import Element, ElementContent, ElementType, Region
from extraction.output import OutputWriter
from extraction.stages.assemble import run_assemble


def _full_seed(out_dir: Path, *, mark_text: bool, mark_fig: bool):
    writer = OutputWriter(out_dir)
    (out_dir / "pages" / "0").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 800)).save(out_dir / "pages" / "0" / "page.png")
    regions = [
        Region(page=0, bbox=[0.0, 0.0, 100.0, 100.0],
               region_type=ElementType.TEXT, confidence=0.9,
               content=ElementContent(text="x")),
    ]
    writer.write_segmentation(
        regions=regions, doc_id="d1", source_file="x.pdf",
        total_pages=1, segmentation_tool="stub_seg",
    )
    writer.mark_stage_done("segment")
    if mark_text:
        writer.mark_stage_done("extract-text")
    if mark_fig:
        writer.mark_stage_done("describe-figures")
    el = Element(
        element_id="abc1234567890def",
        type=ElementType.TEXT, page=0, bbox=[0.0, 0.0, 100.0, 100.0],
        reading_order_index=0, section_path=[], confidence=0.9,
        extractor="stub_text", content=ElementContent(text="x"),
    )
    writer.write_element_sidecar(el)


def _cfg() -> ExtractionConfig:
    return ExtractionConfig(
        renderer="pymupdf", segmenter="pymupdf_text",
        text_extractor="noop", table_extractor="noop",
        formula_extractor="noop", figure_descriptor="noop",
    )


def test_assemble_happy_path(tmp_path: Path):
    out = tmp_path / "doc1"
    _full_seed(out, mark_text=True, mark_fig=True)
    assert run_assemble([out], _cfg()) == 0
    assert (out / ".stages" / "assemble.done").exists()
    cl = json.loads((out / "content_list.json").read_text(encoding="utf-8"))
    assert cl["doc_id"] == "d1"
    assert cl["source_file"] == "x.pdf"
    assert cl["segmentation_tool"] == "stub_seg"
    assert cl["total_pages"] == 1
    assert len(cl["elements"]) == 1
    assert cl["elements"][0]["element_id"] == "abc1234567890def"
    assert cl["elements"][0]["reading_order_index"] == 0


def test_assemble_missing_prereq_writes_error(tmp_path: Path):
    out = tmp_path / "doc1"
    _full_seed(out, mark_text=False, mark_fig=True)
    assert run_assemble([out], _cfg()) == 1
    err = out / ".stages" / "assemble.error"
    assert err.exists()
    assert "extract-text" in err.read_text(encoding="utf-8")


def test_assemble_skips_when_marker_exists(tmp_path: Path):
    out = tmp_path / "doc1"
    _full_seed(out, mark_text=True, mark_fig=True)
    OutputWriter(out).mark_stage_done("assemble")
    assert run_assemble([out], _cfg()) == 0
