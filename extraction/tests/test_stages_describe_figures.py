"""Unit tests for run_figures with stub adapters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from extraction.config import ExtractionConfig
from extraction.models import ElementContent, ElementType, Region
from extraction.output import OutputWriter
from extraction.registry import register_figure_descriptor
from extraction.stages.describe_figures import run_figures


@register_figure_descriptor("stub_fig")
class _StubFig:
    TOOL_NAME = "stub_fig"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def describe(self, image, caption=None):
        return f"a stub description {image.size}"


@register_figure_descriptor("stub_fig_empty")
class _StubFigEmpty:
    TOOL_NAME = "stub_fig_empty"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def describe(self, image, caption=None):
        return ""


@register_figure_descriptor("stub_fig_broken")
class _StubFigBroken:
    TOOL_NAME = "stub_fig_broken"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def describe(self, image, caption=None):
        raise RuntimeError("describe blew up")


def _seed_segment(out_dir: Path) -> None:
    writer = OutputWriter(out_dir)
    (out_dir / "pages" / "0").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 800)).save(out_dir / "pages" / "0" / "page.png")
    regions = [
        Region(page=0, bbox=[0.0, 0.0, 100.0, 100.0],
               region_type=ElementType.FIGURE, confidence=0.9,
               content=ElementContent(caption="Fig 1")),
        Region(page=0, bbox=[0.0, 100.0, 100.0, 200.0],
               region_type=ElementType.DIAGRAM, confidence=0.9),
    ]
    writer.write_segmentation(
        regions=regions, doc_id="d1", source_file="x.pdf",
        total_pages=1, segmentation_tool="stub_segmenter",
    )
    writer.mark_stage_done("segment")


def _cfg(figure_descriptor: str = "stub_fig") -> ExtractionConfig:
    return ExtractionConfig(
        renderer="pymupdf", segmenter="pymupdf_text",
        text_extractor="noop", table_extractor="noop",
        formula_extractor="noop", figure_descriptor=figure_descriptor,
    )


def test_figures_happy_path(tmp_path: Path) -> None:
    out = tmp_path / "doc1"
    _seed_segment(out)
    assert run_figures([out], _cfg()) == 0
    assert (out / ".stages" / "describe-figures.done").exists()
    fig_sidecars = list((out / "pages" / "0").glob("*_figure.json"))
    diag_sidecars = list((out / "pages" / "0").glob("*_diagram.json"))
    assert len(fig_sidecars) == 1
    assert len(diag_sidecars) == 1
    fig = json.loads(fig_sidecars[0].read_text(encoding="utf-8"))
    assert fig["content"]["description"].startswith("a stub description")
    assert fig["content"]["caption"] == "Fig 1"
    assert fig["content"]["image_path"].endswith(".png")


def test_figures_drops_empty_when_no_crop_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If describer returns empty AND crop save fails, the per-PDF errors out."""
    out = tmp_path / "doc1"
    _seed_segment(out)

    def _skip(self, **kw):
        raise RuntimeError("simulated crop failure")
    monkeypatch.setattr(OutputWriter, "save_element_crop", _skip)

    cfg = _cfg(figure_descriptor="stub_fig_empty")
    exit_code = run_figures([out], cfg)
    assert exit_code == 1


def test_figures_skips_when_marker_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "doc1"
    _seed_segment(out)
    OutputWriter(out).mark_stage_done("describe-figures")
    from extraction.stages import describe_figures as mod
    def _boom(*a, **kw):
        raise AssertionError("figure descriptor must not be loaded")
    monkeypatch.setattr(mod, "get_figure_descriptor", _boom)
    assert run_figures([out], _cfg()) == 0


def test_figures_missing_prereq_writes_error(tmp_path: Path) -> None:
    out = tmp_path / "doc1"
    out.mkdir(parents=True)
    assert run_figures([out], _cfg()) == 1
    assert (out / ".stages" / "describe-figures.error").exists()


def test_figures_error_writes_marker_and_continues(tmp_path: Path) -> None:
    out_a = tmp_path / "doc_a"
    out_b = tmp_path / "doc_b"
    _seed_segment(out_a)
    _seed_segment(out_b)
    exit_code = run_figures([out_a, out_b], _cfg(figure_descriptor="stub_fig_broken"))
    assert exit_code == 1
    for out in (out_a, out_b):
        assert (out / ".stages" / "describe-figures.error").exists()


_captured_captions: list[str | None] = []


@register_figure_descriptor("stub_fig_capture")
class _StubFigCapture:
    TOOL_NAME = "stub_fig_capture"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def describe(self, image, caption=None):
        _captured_captions.append(caption)
        return "ok"


def test_figures_passes_caption_to_describer(tmp_path: Path) -> None:
    """Stage 3 reicht region.content.caption durch an describe()."""
    _captured_captions.clear()
    out = tmp_path / "doc1"
    _seed_segment(out)  # seed hat FIGURE mit caption="Fig 1", DIAGRAM ohne
    cfg = _cfg(figure_descriptor="stub_fig_capture")
    assert run_figures([out], cfg) == 0
    assert "Fig 1" in _captured_captions
    assert None in _captured_captions
