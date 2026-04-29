"""Unit tests for run_segment with stub adapters (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from extraction.config import ExtractionConfig
from extraction.models import ElementContent, ElementType, Region
from extraction.registry import (
    register_renderer,
    register_segmenter,
)
from extraction.stages.segment import run_segment


@register_renderer("stub_renderer")
class _StubRenderer:
    TOOL_NAME = "stub_renderer"

    def __init__(self, **_: object) -> None:
        pass

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def page_count(self, pdf_path):
        return 2

    def render_page(self, pdf_path, page_number):
        return Image.new("RGB", (600, 800), "white")

    def render_all(self, pdf_path):
        return [self.render_page(pdf_path, i) for i in range(2)]


@register_segmenter("stub_segmenter")
class _StubSegmenter:
    TOOL_NAME = "stub_segmenter"

    def __init__(self, **_: object) -> None:
        pass

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def segment(self, pdf_path):
        return [
            Region(page=0, bbox=[10.0, 20.0, 100.0, 60.0],
                   region_type=ElementType.TEXT, reading_order_index=0,
                   confidence=0.9, content=ElementContent(text="hello")),
            Region(page=1, bbox=[0.0, 0.0, 200.0, 100.0],
                   region_type=ElementType.TABLE, reading_order_index=1,
                   confidence=0.8,
                   content=ElementContent(markdown="| a | b |\n|---|---|")),
        ]


def _make_cfg() -> ExtractionConfig:
    return ExtractionConfig(
        renderer="stub_renderer",
        segmenter="stub_segmenter",
        text_extractor="noop",
        table_extractor="stub_segmenter",
        formula_extractor="noop",
        figure_descriptor="noop",
    )


def test_segment_happy_path(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% dummy\n")
    cfg = _make_cfg()

    exit_code = run_segment([pdf], cfg, output_base=tmp_path / "outputs")

    out = tmp_path / "outputs" / "sample"
    assert exit_code == 0
    assert (out / ".stages" / "segment.done").exists()
    assert (out / "pages" / "0" / "page.png").exists()
    assert (out / "pages" / "1" / "page.png").exists()
    seg = json.loads((out / "segmentation.json").read_text(encoding="utf-8"))
    assert seg["segmentation_tool"] == "stub_segmenter"
    assert seg["total_pages"] == 2
    assert seg["render_dpi"] == 150
    assert seg["stage_config"]["renderer"] == "stub_renderer"
    assert seg["stage_config"]["segmenter"] == "stub_segmenter"
    assert seg["stage_config"]["table_extractor"] == "stub_segmenter"
    assert len(seg["regions"]) == 2

    table_sidecars = list((out / "pages" / "1").glob("*_table.json"))
    assert len(table_sidecars) == 1
    el = json.loads(table_sidecars[0].read_text(encoding="utf-8"))
    assert el["type"] == "table"
    assert el["content"]["markdown"].startswith("| a | b |")

    assert not list((out / "pages" / "0").glob("*_text.json"))


def test_segment_threads_region_reading_order_index_to_element(tmp_path: Path) -> None:
    """The Element sidecar must carry the segmenter's reading_order_index, not 0."""
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% dummy\n")
    cfg = _make_cfg()

    run_segment([pdf], cfg, output_base=tmp_path / "outputs")

    table_sidecars = list(
        (tmp_path / "outputs" / "sample" / "pages" / "1").glob("*_table.json")
    )
    assert len(table_sidecars) == 1
    el = json.loads(table_sidecars[0].read_text(encoding="utf-8"))
    # _StubSegmenter returns the table as the 2nd region (index 1). Without the
    # fix this defaulted to 0 and collapsed the sort in build_content_list.
    assert el["reading_order_index"] == 1


def test_segment_skips_when_marker_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% dummy\n")
    cfg = _make_cfg()
    assert run_segment([pdf], cfg, output_base=tmp_path / "outputs") == 0

    from extraction.stages import segment as seg_mod

    def _boom(*a, **kw):
        raise AssertionError("segmenter must not be loaded when all paths are skipped")
    monkeypatch.setattr(seg_mod, "get_segmenter", _boom)

    exit_code = run_segment([pdf], cfg, output_base=tmp_path / "outputs")
    assert exit_code == 0


def test_segment_done_with_stale_config_writes_error_without_overwrite(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% dummy\n")
    cfg = _make_cfg()
    output_base = tmp_path / "outputs"
    assert run_segment([pdf], cfg, output_base=output_base) == 0

    out = output_base / "sample"
    page_image = out / "pages" / "0" / "page.png"
    mtime_before = page_image.stat().st_mtime_ns

    stale_cfg = cfg.model_copy(update={"confidence_threshold": 0.7})
    exit_code = run_segment([pdf], stale_cfg, output_base=output_base)

    assert exit_code == 1
    err = out / ".stages" / "segment.error"
    assert err.exists()
    assert "stage_config" in err.read_text(encoding="utf-8")
    assert page_image.stat().st_mtime_ns == mtime_before
    assert not (out / ".stages" / "segment.done").exists()


def test_segment_done_with_changed_pdf_writes_error(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% dummy\n")
    cfg = _make_cfg()
    output_base = tmp_path / "outputs"
    assert run_segment([pdf], cfg, output_base=output_base) == 0

    pdf.write_bytes(b"%PDF-1.4\n% changed\n")
    exit_code = run_segment([pdf], cfg, output_base=output_base)

    assert exit_code == 1
    err = output_base / "sample" / ".stages" / "segment.error"
    assert err.exists()
    assert "doc_id" in err.read_text(encoding="utf-8")


@register_segmenter("stub_broken")
class _BrokenSegmenter:
    TOOL_NAME = "stub_broken"

    def __init__(self, **_: object) -> None:
        pass

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def segment(self, pdf_path):
        raise RuntimeError("segmenter blew up")


def test_segment_dirty_output_dir_without_done_marker_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-existing artefacts without segment.done means a crashed prior run.
    Stage must refuse to re-process and write .error, not silently overwrite."""
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% dummy\n")
    out_dir = tmp_path / "outputs" / "sample"
    (out_dir / "pages" / "0").mkdir(parents=True)
    (out_dir / "pages" / "0" / "page.png").write_bytes(b"")
    (out_dir / "segmentation.json").write_text("{}", encoding="utf-8")
    cfg = _make_cfg()

    from extraction.stages import segment as seg_mod

    def _boom(*a, **kw):
        raise AssertionError("segmenter must not be loaded when output dir is dirty")
    monkeypatch.setattr(seg_mod, "get_segmenter", _boom)

    exit_code = run_segment([pdf], cfg, output_base=tmp_path / "outputs")
    assert exit_code == 1
    err = out_dir / ".stages" / "segment.error"
    assert err.exists()
    assert "already contains artefacts" in err.read_text(encoding="utf-8")
    assert not (out_dir / ".stages" / "segment.done").exists()


def test_segment_error_writes_marker_and_continues(tmp_path: Path) -> None:
    pdf_ok = tmp_path / "good.pdf"
    pdf_ok.write_bytes(b"%PDF-1.4\n")
    pdf_bad = tmp_path / "bad.pdf"
    pdf_bad.write_bytes(b"%PDF-1.4\n")

    cfg = _make_cfg().model_copy(update={"segmenter": "stub_broken"})
    exit_code = run_segment([pdf_bad, pdf_ok], cfg, output_base=tmp_path / "outputs")

    assert exit_code == 1
    for stem in ("good", "bad"):
        err = tmp_path / "outputs" / stem / ".stages" / "segment.error"
        done = tmp_path / "outputs" / stem / ".stages" / "segment.done"
        assert err.exists()
        assert not done.exists()
        assert "segmenter blew up" in err.read_text(encoding="utf-8")
