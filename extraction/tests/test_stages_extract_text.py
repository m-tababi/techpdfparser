"""Unit tests for run_text with stub adapters."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from extraction.config import ExtractionConfig
from extraction.models import ElementContent, ElementType, Region
from extraction.output import OutputWriter
from extraction.registry import register_text_extractor
from extraction.stages.extract_text import run_text


@register_text_extractor("stub_text")
class _StubText:
    TOOL_NAME = "stub_text"

    def __init__(self, **_: object) -> None:
        pass

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def extract(self, page_image, page_number):
        return ElementContent(text=f"extracted page {page_number}")


def _seed_segment(out_dir: Path) -> None:
    writer = OutputWriter(out_dir)
    (out_dir / "pages" / "0").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 800)).save(out_dir / "pages" / "0" / "page.png")
    regions = [
        Region(page=0, bbox=[10.0, 20.0, 100.0, 60.0],
               region_type=ElementType.TEXT, confidence=0.9,
               content=ElementContent(text="original")),
        Region(page=0, bbox=[0.0, 0.0, 200.0, 40.0],
               region_type=ElementType.HEADING, confidence=0.95,
               content=ElementContent(text="a title")),
    ]
    writer.write_segmentation(
        regions=regions, doc_id="d1", source_file="x.pdf",
        total_pages=1, segmentation_tool="stub_segmenter",
    )
    writer.mark_stage_done("segment")


def _cfg() -> ExtractionConfig:
    return ExtractionConfig(
        renderer="pymupdf",
        segmenter="pymupdf_text",
        text_extractor="stub_text",
        table_extractor="noop",
        formula_extractor="noop",
        figure_descriptor="noop",
    )


def test_text_happy_path(tmp_path: Path):
    out = tmp_path / "doc1"
    _seed_segment(out)
    exit_code = run_text([out], _cfg())
    assert exit_code == 0
    assert (out / ".stages" / "extract-text.done").exists()
    text_sidecars = list((out / "pages" / "0").glob("*_text.json"))
    heading_sidecars = list((out / "pages" / "0").glob("*_heading.json"))
    assert len(text_sidecars) == 1
    assert len(heading_sidecars) == 1
    text_el = json.loads(text_sidecars[0].read_text(encoding="utf-8"))
    assert text_el["content"]["text"] == "extracted page 0"
    assert text_el["extractor"] == "stub_text"


def test_text_skips_when_marker_exists(tmp_path: Path, monkeypatch):
    out = tmp_path / "doc1"
    _seed_segment(out)
    OutputWriter(out).mark_stage_done("extract-text")

    from extraction.stages import extract_text as mod
    def _boom(*a, **kw):
        raise AssertionError("text extractor must not be loaded")
    monkeypatch.setattr(mod, "get_text_extractor", _boom)

    assert run_text([out], _cfg()) == 0


def test_text_missing_prereq_writes_error(tmp_path: Path):
    out = tmp_path / "doc1"
    out.mkdir(parents=True)
    assert run_text([out], _cfg()) == 1
    err = out / ".stages" / "extract-text.error"
    assert err.exists()
    assert "segment" in err.read_text(encoding="utf-8")


@register_text_extractor("stub_text_broken")
class _BrokenText:
    TOOL_NAME = "stub_text_broken"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def extract(self, page_image, page_number):
        raise RuntimeError("ocr blew up")


def test_text_error_writes_marker_and_continues(tmp_path: Path):
    out_a = tmp_path / "doc_a"
    out_b = tmp_path / "doc_b"
    _seed_segment(out_a)
    _seed_segment(out_b)
    cfg = _cfg().model_copy(update={"text_extractor": "stub_text_broken"})
    exit_code = run_text([out_a, out_b], cfg)
    assert exit_code == 1
    for out in (out_a, out_b):
        assert (out / ".stages" / "extract-text.error").exists()
        assert not (out / ".stages" / "extract-text.done").exists()
