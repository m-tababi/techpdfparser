"""Unit tests for run_text with stub adapters."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import extraction.adapters  # noqa: F401 — trigger noop adapter registration
from extraction.config import ExtractionConfig
from extraction.models import ElementContent, ElementType, Region
from extraction.output import OutputWriter
from extraction.registry import register_table_extractor, register_text_extractor
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


def test_text_skips_region_when_sidecar_already_exists(tmp_path: Path, monkeypatch):
    """Region mit existierendem Sidecar (Stage-1-Passthrough) wird nicht re-extracted."""
    out = tmp_path / "doc1"
    _seed_segment(out)

    # Existing sidecar simulieren (als hätte Stage 1 das im Role-Match-Pfad
    # geschrieben). Sidecar-Name: <element_id>_text.json im pages/0/.
    import hashlib
    meta = OutputWriter(out).read_segmentation()
    text_region = next(r for r in meta["regions"] if r.region_type == ElementType.TEXT)
    x0, y0, x1, y1 = (round(v) for v in text_region.bbox)
    raw = f"{meta['doc_id']}:{text_region.page}:{text_region.region_type.value}:{x0},{y0},{x1},{y1}"
    el_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
    sidecar = out / "pages" / "0" / f"{el_id}_text.json"
    sidecar.write_text(
        '{"element_id":"' + el_id + '","type":"text","page":0,'
        '"bbox":[10.0,20.0,100.0,60.0],"reading_order_index":0,'
        '"section_path":[],"confidence":0.9,"extractor":"stub_segmenter",'
        '"content":{"text":"from stage 1 passthrough"}}',
        encoding="utf-8",
    )
    mtime_before = sidecar.stat().st_mtime_ns

    exit_code = run_text([out], _cfg())
    assert exit_code == 0
    # Sidecar darf nicht überschrieben worden sein.
    assert sidecar.stat().st_mtime_ns == mtime_before
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["extractor"] == "stub_segmenter"
    assert data["content"]["text"] == "from stage 1 passthrough"


_captured_text_sizes: list[tuple[int, int]] = []


@register_text_extractor("stub_text_capture")
class _StubTextCapture:
    TOOL_NAME = "stub_text_capture"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def extract(self, image, page_number):
        _captured_text_sizes.append(image.size)
        return ElementContent(text="ok")


def test_text_extractor_receives_region_crop(tmp_path: Path):
    """TextExtractor bekommt Region-Crop, nicht die Vollseite."""
    _captured_text_sizes.clear()
    out = tmp_path / "doc1"
    _seed_segment(out)
    cfg = _cfg().model_copy(update={"text_extractor": "stub_text_capture"})
    assert run_text([out], cfg) == 0
    assert _captured_text_sizes, "extractor nicht aufgerufen"
    for w, h in _captured_text_sizes:
        assert (w, h) != (600, 800), (
            f"extractor bekam Vollseite {(w, h)}, nicht Crop"
        )


@register_table_extractor("stub_table")
class _StubTable:
    TOOL_NAME = "stub_table"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def extract(self, region_image, page_number):
        return ElementContent(
            markdown="| h1 | h2 |\n| --- | --- |\n| a | b |",
            text="| h1 | h2 |\n| --- | --- |\n| a | b |",
            html="<table><tr><td>h1</td></tr></table>",
        )


def _seed_segment_with_table(out_dir: Path) -> None:
    writer = OutputWriter(out_dir)
    (out_dir / "pages" / "0").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 800)).save(out_dir / "pages" / "0" / "page.png")
    regions = [
        Region(page=0, bbox=[10.0, 10.0, 200.0, 100.0],
               region_type=ElementType.TABLE, confidence=0.95,
               content=ElementContent(caption="Table 1. Demo.")),
    ]
    writer.write_segmentation(
        regions=regions, doc_id="d1", source_file="x.pdf",
        total_pages=1, segmentation_tool="stub_segmenter",
    )
    writer.mark_stage_done("segment")


def test_table_role_mismatch_extracts_sidecar(tmp_path: Path):
    """Bei table_extractor != segmenter ruft Stage 2 den Table-Extractor auf."""
    out = tmp_path / "doc1"
    _seed_segment_with_table(out)
    cfg = _cfg().model_copy(update={"table_extractor": "stub_table"})
    assert run_text([out], cfg) == 0
    table_sidecars = list((out / "pages" / "0").glob("*_table.json"))
    assert len(table_sidecars) == 1
    data = json.loads(table_sidecars[0].read_text(encoding="utf-8"))
    assert data["extractor"] == "stub_table"
    assert data["content"]["markdown"].startswith("| h1")
    assert data["content"]["caption"] == "Table 1. Demo."


from extraction.registry import register_formula_extractor


@register_formula_extractor("stub_formula")
class _StubFormula:
    TOOL_NAME = "stub_formula"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def extract(self, region_image, page_number):
        return ElementContent(latex="E = mc^2", text="E = mc^2")


def _seed_segment_with_formula(out_dir: Path) -> None:
    writer = OutputWriter(out_dir)
    (out_dir / "pages" / "0").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 800)).save(out_dir / "pages" / "0" / "page.png")
    regions = [
        Region(page=0, bbox=[50.0, 50.0, 150.0, 80.0],
               region_type=ElementType.FORMULA, confidence=0.95),
    ]
    writer.write_segmentation(
        regions=regions, doc_id="d1", source_file="x.pdf",
        total_pages=1, segmentation_tool="stub_segmenter",
    )
    writer.mark_stage_done("segment")


def test_formula_role_mismatch_extracts_sidecar(tmp_path: Path):
    out = tmp_path / "doc1"
    _seed_segment_with_formula(out)
    cfg = _cfg().model_copy(update={"formula_extractor": "stub_formula"})
    assert run_text([out], cfg) == 0
    formula_sidecars = list((out / "pages" / "0").glob("*_formula.json"))
    assert len(formula_sidecars) == 1
    data = json.loads(formula_sidecars[0].read_text(encoding="utf-8"))
    assert data["content"]["latex"] == "E = mc^2"
    assert data["extractor"] == "stub_formula"


@register_table_extractor("stub_table_empty")
class _StubTableEmpty:
    TOOL_NAME = "stub_table_empty"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def extract(self, region_image, page_number):
        return ElementContent()


def test_table_persists_with_image_path_when_content_empty(tmp_path: Path):
    """Table mit leerem Extractor-Output persistiert, solange Crop + image_path da sind."""
    out = tmp_path / "doc1"
    _seed_segment_with_table(out)
    cfg = _cfg().model_copy(update={"table_extractor": "stub_table_empty"})
    assert run_text([out], cfg) == 0
    table_sidecars = list((out / "pages" / "0").glob("*_table.json"))
    crops = list((out / "pages" / "0").glob("*_table.png"))
    assert len(table_sidecars) == 1
    assert len(crops) == 1
    data = json.loads(table_sidecars[0].read_text(encoding="utf-8"))
    assert data["content"]["image_path"].endswith("_table.png")
    assert data["content"]["caption"] == "Table 1. Demo."
