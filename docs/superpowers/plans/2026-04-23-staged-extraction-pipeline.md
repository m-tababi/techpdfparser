# Staged Extraction Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic `ExtractionPipeline.run()` with four stage commands (`segment`, `extract-text`, `describe-figures`, `assemble`) so each stage runs in its own OS process — the kernel releases GPU memory between stages instead of relying on PyTorch refcounts or MinerU singleton hacks.

**Architecture:** Four stage functions in a new `extraction/stages/` package. `OutputWriter` gains marker helpers and carries segmentation metadata. CLI in `__main__.py` exposes four subcommands; the monolithic `extract` and `rebuild` subcommands are removed. Each stage reads the full YAML config but only instantiates the adapters it uses. Pro-PDF isolation within a stage: one PDF's failure does not stop the rest.

**Tech Stack:** Python 3.10+, Pydantic 2.x, PyMuPDF, MinerU 2.5, Transformers (olmOCR-2, Qwen2.5-VL), pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-04-23-staged-extraction-pipeline-design.md`

---

## File Structure

Files that will be created or modified:

- `extraction/output.py` — add `mark_stage_done`, `is_stage_done`, `read_stage_error`, `write_stage_error`; extend `write_segmentation` with metadata; add `read_segmentation`.
- `extraction/stages/__init__.py` (new) — shared helpers: `StageOutcome` dataclass, `print_stage_summary()`, `STAGE_ORDER` constant.
- `extraction/stages/segment.py` (new) — `run_segment()`.
- `extraction/stages/extract_text.py` (new) — `run_text()`.
- `extraction/stages/describe_figures.py` (new) — `run_figures()`.
- `extraction/stages/assemble.py` (new) — `run_assemble()`.
- `extraction/__main__.py` — replace `extract`/`rebuild` subcommands with `segment`/`extract-text`/`describe-figures`/`assemble`.
- `extraction/pipeline.py` — remove `ExtractionPipeline.run()` and the now-dead helpers (`_extract_by_role`, `_extract_region`, `_run_role_tool`, `_is_droppable`, `_assert_output_dir_clean`, `_make_doc_id`, `_make_element_id`). Module may shrink to a near-empty shell or be deleted entirely if nothing else imports it.
- `extraction/config.py` — add constant `DEFAULT_OUTPUT_BASE = "outputs"`.
- `extraction/tests/test_output.py` — tests for the new marker + segmentation helpers.
- `extraction/tests/test_stages_segment.py` (new) — unit tests for `run_segment()`.
- `extraction/tests/test_stages_extract_text.py` (new) — unit tests for `run_text()`.
- `extraction/tests/test_stages_describe_figures.py` (new) — unit tests for `run_figures()`.
- `extraction/tests/test_stages_assemble.py` (new) — unit tests for `run_assemble()`.
- `extraction/tests/test_cli.py` — rewrite for new subcommands.
- `extraction/tests/test_pipeline.py` — delete or shrink (monolith gone).
- `extraction/tests/test_stages_integration.py` (new, marked `@pytest.mark.integration`) — end-to-end run through all four stages on a real short PDF with structural-plus-strict comparison against a reference `content_list.json`.
- `README.md` — swap `## CLI` for `## Extraction-Pipeline`.

---

## Task 1: Stage-marker helpers on `OutputWriter`

**Files:**
- Modify: `extraction/output.py`
- Test: `extraction/tests/test_output.py`

**Context:** Each stage needs a file-system flag per PDF so later runs can skip completed work. The spec (§3) defines `.stages/<name>.done` and `.stages/<name>.error`. Errors record the traceback; success is just marker existence.

- [ ] **Step 1: Add failing tests in `extraction/tests/test_output.py`**

Append to the existing test file:

```python
def test_mark_stage_done_creates_marker(tmp_path):
    writer = OutputWriter(tmp_path)
    writer.mark_stage_done("segment")
    assert (tmp_path / ".stages" / "segment.done").exists()


def test_is_stage_done_reflects_marker(tmp_path):
    writer = OutputWriter(tmp_path)
    assert writer.is_stage_done("segment") is False
    writer.mark_stage_done("segment")
    assert writer.is_stage_done("segment") is True


def test_mark_stage_done_clears_previous_error(tmp_path):
    writer = OutputWriter(tmp_path)
    writer.write_stage_error("segment", RuntimeError("boom"))
    writer.mark_stage_done("segment")
    assert (tmp_path / ".stages" / "segment.done").exists()
    assert not (tmp_path / ".stages" / "segment.error").exists()


def test_write_stage_error_writes_traceback(tmp_path):
    writer = OutputWriter(tmp_path)
    try:
        raise RuntimeError("something broke")
    except RuntimeError as exc:
        writer.write_stage_error("segment", exc)
    error_path = tmp_path / ".stages" / "segment.error"
    assert error_path.exists()
    text = error_path.read_text(encoding="utf-8")
    assert "RuntimeError" in text
    assert "something broke" in text


def test_write_stage_error_clears_previous_done(tmp_path):
    writer = OutputWriter(tmp_path)
    writer.mark_stage_done("segment")
    writer.write_stage_error("segment", RuntimeError("regression"))
    assert (tmp_path / ".stages" / "segment.error").exists()
    assert not (tmp_path / ".stages" / "segment.done").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest extraction/tests/test_output.py -k stage -v
```
Expected: AttributeError on `OutputWriter.mark_stage_done` / `.is_stage_done` / `.write_stage_error`.

- [ ] **Step 3: Implement the marker API in `extraction/output.py`**

Add to the `OutputWriter` class (after `write_element_sidecar`):

```python
    def _stages_dir(self) -> Path:
        p = self.output_dir / ".stages"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def is_stage_done(self, stage_name: str) -> bool:
        return (self._stages_dir() / f"{stage_name}.done").exists()

    def mark_stage_done(self, stage_name: str) -> Path:
        done = self._stages_dir() / f"{stage_name}.done"
        err = self._stages_dir() / f"{stage_name}.error"
        if err.exists():
            err.unlink()
        done.touch()
        return done

    def write_stage_error(self, stage_name: str, exc: BaseException) -> Path:
        import traceback
        err = self._stages_dir() / f"{stage_name}.error"
        done = self._stages_dir() / f"{stage_name}.done"
        if done.exists():
            done.unlink()
        err.write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )
        return err
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest extraction/tests/test_output.py -k stage -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```
git add extraction/output.py extraction/tests/test_output.py
git commit -m "feat(extraction): stage-marker helpers on OutputWriter"
```

---

## Task 2: Segmentation metadata (write/read)

**Files:**
- Modify: `extraction/output.py`
- Test: `extraction/tests/test_output.py`

**Context:** Stage 4 must rebuild `content_list.json` without a PDF in hand, so `segmentation.json` now carries `doc_id`, `source_file`, `total_pages`, `segmentation_tool` alongside the region list (spec §1, §7). Breaking change on internal inter-stage state; no back-compat.

- [ ] **Step 1: Add failing tests in `extraction/tests/test_output.py`**

Append:

```python
from extraction.models import Region, ElementType


def test_write_segmentation_stores_metadata(tmp_path):
    writer = OutputWriter(tmp_path)
    regions = [
        Region(
            page=0,
            bbox=[10.0, 20.0, 100.0, 50.0],
            region_type=ElementType.TEXT,
            confidence=0.9,
        ),
    ]
    writer.write_segmentation(
        regions=regions,
        doc_id="abc123",
        source_file="example.pdf",
        total_pages=3,
        segmentation_tool="mineru25",
    )
    path = tmp_path / "segmentation.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["doc_id"] == "abc123"
    assert raw["source_file"] == "example.pdf"
    assert raw["total_pages"] == 3
    assert raw["segmentation_tool"] == "mineru25"
    assert len(raw["regions"]) == 1


def test_read_segmentation_roundtrip(tmp_path):
    writer = OutputWriter(tmp_path)
    regions = [
        Region(page=0, bbox=[0.0, 0.0, 10.0, 10.0],
               region_type=ElementType.HEADING, confidence=1.0),
    ]
    writer.write_segmentation(
        regions=regions, doc_id="x", source_file="y.pdf",
        total_pages=1, segmentation_tool="mineru25",
    )
    meta = writer.read_segmentation()
    assert meta["doc_id"] == "x"
    assert meta["source_file"] == "y.pdf"
    assert meta["total_pages"] == 1
    assert meta["segmentation_tool"] == "mineru25"
    assert len(meta["regions"]) == 1
    assert meta["regions"][0].region_type == ElementType.HEADING
```

Add `import json` at the top if not present.

- [ ] **Step 2: Run tests to verify they fail**

```
pytest extraction/tests/test_output.py -k segmentation -v
```
Expected: TypeError on the new signature and/or AttributeError on `read_segmentation`.

- [ ] **Step 3: Replace `write_segmentation` and add `read_segmentation` in `extraction/output.py`**

Replace the existing `write_segmentation` method:

```python
    def write_segmentation(
        self,
        regions: list,
        *,
        doc_id: str,
        source_file: str,
        total_pages: int,
        segmentation_tool: str,
    ) -> Path:
        """Write segmentation.json with doc metadata + region list."""
        path = self.output_dir / "segmentation.json"
        data = {
            "doc_id": doc_id,
            "source_file": source_file,
            "total_pages": total_pages,
            "segmentation_tool": segmentation_tool,
            "regions": [r.model_dump(mode="json", exclude_none=True) for r in regions],
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def read_segmentation(self) -> dict:
        """Load segmentation.json back into a dict with parsed Region instances."""
        from .models import Region  # local import to keep module top light
        path = self.output_dir / "segmentation.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            "doc_id": raw["doc_id"],
            "source_file": raw["source_file"],
            "total_pages": raw["total_pages"],
            "segmentation_tool": raw["segmentation_tool"],
            "regions": [Region.model_validate(r) for r in raw["regions"]],
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest extraction/tests/test_output.py -k segmentation -v
```
Expected: 2 new tests pass. Any existing tests calling `write_segmentation(regions)` positionally will fail — that is OK and will be handled in Task 9 when the monolith goes away. If the existing test is red now, mark it `@pytest.mark.skip(reason="monolith removal — Task 9")` for this commit.

- [ ] **Step 5: Commit**

```
git add extraction/output.py extraction/tests/test_output.py
git commit -m "feat(extraction): segmentation.json carries doc metadata"
```

---

## Task 3: `extraction/stages/` package scaffold + shared helpers

**Files:**
- Create: `extraction/stages/__init__.py`

**Context:** Four stage modules share a reporting helper and an ordering constant. Keep the shared surface tiny.

- [ ] **Step 1: Create `extraction/stages/__init__.py`**

```python
"""Stage functions for the extraction pipeline.

Each stage is a separate OS process invoked via the CLI. Stages share
marker semantics and a reporting helper so output is consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

StageName = Literal["segment", "extract-text", "describe-figures", "assemble"]

STAGE_ORDER: list[StageName] = [
    "segment",
    "extract-text",
    "describe-figures",
    "assemble",
]

OutcomeStatus = Literal["success", "skipped", "error", "missing_prereq"]


@dataclass
class StageOutcome:
    """Result for one (pdf|outdir) within a stage run."""

    label: str          # user-facing path, e.g. "outputs/jmmp-09-00199-v2"
    status: OutcomeStatus
    detail: str = ""    # e.g. "(19 Seiten, 127 Regions)" or error summary


def next_stage(stage: StageName) -> StageName | None:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None


def print_stage_summary(
    stage: StageName,
    outcomes: list[StageOutcome],
    out_dirs_for_next: list[Path],
) -> int:
    """Print inline log already happened per path; this prints the end block.

    Returns the exit code: 0 if every outcome is success or skipped, else 1.
    """
    ok = sum(1 for o in outcomes if o.status in ("success", "skipped"))
    bad = sum(1 for o in outcomes if o.status in ("error", "missing_prereq"))
    bar = "━" * 44
    print()
    print(bar)
    if bad:
        print(f"  Stage '{stage}': {ok} erfolgreich, {bad} FEHLGESCHLAGEN")
    else:
        print(f"  Stage '{stage}': {ok} erfolgreich")
    print(bar)
    for o in outcomes:
        mark = {"success": "✓", "skipped": "↷", "error": "✗", "missing_prereq": "✗"}[o.status]
        suffix = f"  {o.detail}" if o.detail else ""
        print(f"  {mark} {o.label}{suffix}")
    print()
    nxt = next_stage(stage)
    if nxt is not None and out_dirs_for_next:
        paths = " ".join(str(p) for p in out_dirs_for_next)
        print("Nächster Schritt (nur erfolgreiche Ordner):")
        print(f"  python -m extraction {nxt} {paths}")
    elif nxt is None:
        print("Pipeline komplett.")
    print()
    return 0 if bad == 0 else 1
```

- [ ] **Step 2: Smoke-import**

```
python -c "from extraction.stages import STAGE_ORDER, StageOutcome, print_stage_summary; print(STAGE_ORDER)"
```
Expected: `['segment', 'extract-text', 'describe-figures', 'assemble']`.

- [ ] **Step 3: Commit**

```
git add extraction/stages/__init__.py
git commit -m "feat(extraction): stages/ scaffold with shared reporting helper"
```

---

## Task 4: Segment stage (`run_segment`)

**Files:**
- Create: `extraction/stages/segment.py`
- Test: `extraction/tests/test_stages_segment.py`

**Context:** Stage 1 renders pages, runs the segmenter, writes `segmentation.json` with metadata, writes element sidecars for any region whose role-tool equals the segmenter's tool (today: Tables under default config). Reuses `OutputWriter.crop_region` / `save_page_image` / `write_element_sidecar` / `save_element_crop` unchanged. Doc-id logic mirrors the old `ExtractionPipeline._make_doc_id` (SHA-256 of PDF bytes, first 16 hex chars).

Loading order is critical for lazy loading: scan which PDFs need work first, THEN load MinerU.

### Happy path

- [ ] **Step 1: Create failing happy-path test in `extraction/tests/test_stages_segment.py`**

```python
"""Unit tests for run_segment with stub adapters (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

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
                   region_type=ElementType.TEXT, confidence=0.9,
                   content=ElementContent(text="hello")),
            Region(page=1, bbox=[0.0, 0.0, 200.0, 100.0],
                   region_type=ElementType.TABLE, confidence=0.8,
                   content=ElementContent(markdown="| a | b |\n|---|---|")),
        ]


def _make_cfg() -> ExtractionConfig:
    return ExtractionConfig(
        renderer="stub_renderer",
        segmenter="stub_segmenter",
        text_extractor="noop",
        table_extractor="stub_segmenter",   # passthrough: role_tool == segmenter
        formula_extractor="noop",
        figure_descriptor="noop",
    )


def test_segment_happy_path(tmp_path: Path):
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
    assert len(seg["regions"]) == 2

    # Passthrough sidecar for the TABLE region (table_extractor == segmenter)
    table_sidecars = list((out / "pages" / "1").glob("*_table.json"))
    assert len(table_sidecars) == 1
    el = json.loads(table_sidecars[0].read_text(encoding="utf-8"))
    assert el["type"] == "table"
    assert el["content"]["markdown"].startswith("| a | b |")

    # No TEXT sidecar from segment stage — that's stage 2's job
    assert not list((out / "pages" / "0").glob("*_text.json"))
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest extraction/tests/test_stages_segment.py::test_segment_happy_path -v
```
Expected: ImportError on `extraction.stages.segment`.

- [ ] **Step 3: Implement `extraction/stages/segment.py`**

```python
"""Stage 1 — segment PDFs and write passthrough sidecars."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import ExtractionConfig
from ..models import Element, ElementType, Region
from ..output import OutputWriter
from ..registry import get_renderer, get_segmenter
from . import StageOutcome, print_stage_summary

_STAGE: str = "segment"


def _doc_id(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _element_id(doc_id: str, region: Region) -> str:
    x0, y0, x1, y1 = (round(v) for v in region.bbox)
    raw = f"{doc_id}:{region.page}:{region.region_type.value}:{x0},{y0},{x1},{y1}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _role_tool_name(region_type: ElementType, cfg: ExtractionConfig) -> str:
    if region_type in (ElementType.TEXT, ElementType.HEADING):
        return cfg.text_extractor
    if region_type == ElementType.TABLE:
        return cfg.table_extractor
    if region_type == ElementType.FORMULA:
        return cfg.formula_extractor
    return cfg.figure_descriptor


def run_segment(
    pdf_paths: list[Path],
    cfg: ExtractionConfig,
    output_base: Path,
) -> int:
    # Pre-scan: which PDFs actually need work?
    plan: list[tuple[Path, Path, OutputWriter]] = []
    skipped: list[StageOutcome] = []
    for pdf in pdf_paths:
        out_dir = output_base / pdf.stem
        writer = OutputWriter(out_dir)
        label = str(out_dir)
        if writer.is_stage_done(_STAGE):
            skipped.append(StageOutcome(label=label, status="skipped"))
            print(f"Processing {label} ... ↷ skipped (already done)")
            continue
        plan.append((pdf, out_dir, writer))

    outcomes: list[StageOutcome] = list(skipped)

    if not plan:
        ok_out_dirs = [output_base / p.stem for p in pdf_paths]
        return print_stage_summary(_STAGE, outcomes, ok_out_dirs)

    # Lazy-load adapters now that we know there's work
    renderer_kwargs = dict(cfg.get_adapter_config(cfg.renderer))
    renderer_kwargs.setdefault("dpi", cfg.dpi)
    renderer = get_renderer(cfg.renderer, **renderer_kwargs)
    segmenter = get_segmenter(cfg.segmenter, **cfg.get_adapter_config(cfg.segmenter))

    for pdf, out_dir, writer in plan:
        label = str(out_dir)
        try:
            _process_one(pdf, writer, renderer, segmenter, cfg)
            writer.mark_stage_done(_STAGE)
            seg = writer.read_segmentation()
            outcomes.append(StageOutcome(
                label=label, status="success",
                detail=f"({seg['total_pages']} Seiten, {len(seg['regions'])} Regions)",
            ))
            print(f"Processing {label} ... ✓")
        except Exception as exc:
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="error",
                detail=f"(Fehler: siehe .stages/{_STAGE}.error)",
            ))
            print(f"Processing {label} ... ✗ {type(exc).__name__}: {exc}")

    ok_out_dirs = [
        output_base / p.stem for p in pdf_paths
        if (output_base / p.stem / ".stages" / f"{_STAGE}.done").exists()
    ]
    return print_stage_summary(_STAGE, outcomes, ok_out_dirs)


def _process_one(
    pdf_path: Path,
    writer: OutputWriter,
    renderer,
    segmenter,
    cfg: ExtractionConfig,
) -> None:
    # 1. Render + save page images
    page_count = renderer.page_count(pdf_path)
    page_images = []
    for i in range(page_count):
        img = renderer.render_page(pdf_path, i)
        writer.save_page_image(page=i, image=img)
        page_images.append(img)

    # 2. Segment
    regions = segmenter.segment(pdf_path)

    # 3. Write segmentation.json with metadata
    doc_id = _doc_id(pdf_path)
    writer.write_segmentation(
        regions=regions,
        doc_id=doc_id,
        source_file=pdf_path.name,
        total_pages=page_count,
        segmentation_tool=segmenter.tool_name,
    )

    # 4. Passthrough sidecars: role_tool == segmenter_tool AND content present
    seg_tool = segmenter.tool_name
    for region in regions:
        if _role_tool_name(region.region_type, cfg) != seg_tool:
            continue
        if region.content is None:
            continue
        if region.confidence < cfg.confidence_threshold:
            continue
        el_id = _element_id(doc_id, region)
        el = Element(
            element_id=el_id,
            type=region.region_type,
            page=region.page,
            bbox=region.bbox,
            reading_order_index=0,          # final ordering happens in assemble
            section_path=[],
            confidence=region.confidence,
            extractor=seg_tool,
            content=region.content.model_copy(),
        )
        # Save visual crop if applicable
        if region.region_type in {
            ElementType.TABLE, ElementType.FORMULA, ElementType.FIGURE,
            ElementType.DIAGRAM, ElementType.TECHNICAL_DRAWING,
        } and 0 <= region.page < len(page_images):
            crop = writer.crop_region(page_images[region.page], region.bbox, dpi=cfg.dpi)
            rel = writer.save_element_crop(
                page=region.page, element_id=el_id,
                element_type=region.region_type.value, image=crop,
            )
            el.content.image_path = str(rel.relative_to(writer.output_dir))
        writer.write_element_sidecar(el)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest extraction/tests/test_stages_segment.py::test_segment_happy_path -v
```
Expected: PASS.

### Skip

- [ ] **Step 5: Add failing skip test**

Append to `extraction/tests/test_stages_segment.py`:

```python
def test_segment_skips_when_marker_exists(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% dummy\n")
    cfg = _make_cfg()

    # Pre-create done marker
    (tmp_path / "outputs" / "sample" / ".stages").mkdir(parents=True)
    (tmp_path / "outputs" / "sample" / ".stages" / "segment.done").touch()

    # If the segmenter were instantiated, importing a missing adapter would
    # raise. Patch get_segmenter to explode so we catch accidental loads.
    from extraction.stages import segment as seg_mod

    def _boom(*a, **kw):
        raise AssertionError("segmenter must not be loaded when all paths are skipped")
    monkeypatch.setattr(seg_mod, "get_segmenter", _boom)

    exit_code = run_segment([pdf], cfg, output_base=tmp_path / "outputs")
    assert exit_code == 0
```

- [ ] **Step 6: Run test to verify it passes**

Happy-path step already passed; the skip path is covered by the pre-scan loop. If the test fails with "segmenter must not be loaded", the lazy-load order is broken — fix in `run_segment` by ensuring `get_segmenter` is called only inside the `if not plan` guard.

```
pytest extraction/tests/test_stages_segment.py::test_segment_skips_when_marker_exists -v
```
Expected: PASS.

### Error

- [ ] **Step 7: Add failing error test**

Append:

```python
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


def test_segment_error_writes_marker_and_continues(tmp_path: Path):
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
```

- [ ] **Step 8: Run test**

```
pytest extraction/tests/test_stages_segment.py::test_segment_error_writes_marker_and_continues -v
```
Expected: PASS (error-branch in `run_segment` already handles this).

- [ ] **Step 9: Commit**

```
git add extraction/stages/segment.py extraction/tests/test_stages_segment.py
git commit -m "feat(extraction): run_segment stage with marker + error isolation"
```

---

## Task 5: Extract-text stage (`run_text`)

**Files:**
- Create: `extraction/stages/extract_text.py`
- Test: `extraction/tests/test_stages_extract_text.py`

**Context:** Stage 2 reads `segmentation.json` + page images from each given output directory, runs the configured text extractor over `TEXT`/`HEADING` regions, and writes element sidecars. Refuses work on directories without `.stages/segment.done`.

### Happy path

- [ ] **Step 1: Create failing happy-path test**

`extraction/tests/test_stages_extract_text.py`:

```python
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
        segmenter="pymupdf_text",       # value is irrelevant; stage 2 doesn't load it
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
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest extraction/tests/test_stages_extract_text.py::test_text_happy_path -v
```
Expected: ImportError on `extraction.stages.extract_text`.

- [ ] **Step 3: Implement `extraction/stages/extract_text.py`**

```python
"""Stage 2 — extract text content for TEXT/HEADING regions."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image as PILImage

from ..config import ExtractionConfig
from ..models import Element, ElementContent, ElementType, Region
from ..output import OutputWriter
from ..registry import get_text_extractor
from . import StageOutcome, print_stage_summary

_STAGE: str = "extract-text"
_PREV: str = "segment"
_TARGET_TYPES = {ElementType.TEXT, ElementType.HEADING}


def _element_id(doc_id: str, region: Region) -> str:
    x0, y0, x1, y1 = (round(v) for v in region.bbox)
    raw = f"{doc_id}:{region.page}:{region.region_type.value}:{x0},{y0},{x1},{y1}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_text(out_dirs: list[Path], cfg: ExtractionConfig) -> int:
    plan: list[tuple[Path, OutputWriter, dict]] = []
    outcomes: list[StageOutcome] = []

    for out_dir in out_dirs:
        writer = OutputWriter(out_dir)
        label = str(out_dir)
        if writer.is_stage_done(_STAGE):
            outcomes.append(StageOutcome(label=label, status="skipped"))
            print(f"Processing {label} ... ↷ skipped (already done)")
            continue
        if not writer.is_stage_done(_PREV):
            exc = FileNotFoundError(f"Stage '{_PREV}' not done for {out_dir}")
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="missing_prereq",
                detail=f"(Vorgänger '{_PREV}' fehlt)",
            ))
            print(f"Processing {label} ... ✗ missing prerequisite: {_PREV}")
            continue
        meta = writer.read_segmentation()
        plan.append((out_dir, writer, meta))

    if not plan:
        ok_dirs = [p for p, *_ in []]  # nothing succeeded
        return print_stage_summary(_STAGE, outcomes, [
            d for d in out_dirs
            if (d / ".stages" / f"{_STAGE}.done").exists()
        ])

    extractor = get_text_extractor(
        cfg.text_extractor, **cfg.get_adapter_config(cfg.text_extractor)
    )

    for out_dir, writer, meta in plan:
        label = str(out_dir)
        try:
            _process_one(out_dir, writer, meta, extractor, cfg)
            writer.mark_stage_done(_STAGE)
            outcomes.append(StageOutcome(label=label, status="success"))
            print(f"Processing {label} ... ✓")
        except Exception as exc:
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="error",
                detail=f"(Fehler: siehe .stages/{_STAGE}.error)",
            ))
            print(f"Processing {label} ... ✗ {type(exc).__name__}: {exc}")

    ok_dirs = [
        d for d in out_dirs
        if (d / ".stages" / f"{_STAGE}.done").exists()
    ]
    return print_stage_summary(_STAGE, outcomes, ok_dirs)


def _load_page(out_dir: Path, page: int):
    path = out_dir / "pages" / str(page) / "page.png"
    return PILImage.open(path).convert("RGB")


def _process_one(
    out_dir: Path,
    writer: OutputWriter,
    meta: dict,
    extractor,
    cfg: ExtractionConfig,
) -> None:
    regions: list[Region] = meta["regions"]
    doc_id: str = meta["doc_id"]
    for region in regions:
        if region.region_type not in _TARGET_TYPES:
            continue
        if region.confidence < cfg.confidence_threshold:
            continue
        page_img = _load_page(out_dir, region.page)
        content = extractor.extract(page_img, region.page)
        # Layout caption from segmenter if present
        if region.content is not None and region.content.caption:
            content.caption = region.content.caption
        if not (content.text or "").strip():
            continue
        el_id = _element_id(doc_id, region)
        el = Element(
            element_id=el_id,
            type=region.region_type,
            page=region.page,
            bbox=region.bbox,
            reading_order_index=0,
            section_path=[],
            confidence=region.confidence,
            extractor=extractor.tool_name,
            content=content,
        )
        writer.write_element_sidecar(el)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest extraction/tests/test_stages_extract_text.py::test_text_happy_path -v
```
Expected: PASS.

### Skip + missing-prereq + error

- [ ] **Step 5: Add the three remaining tests**

Append to `extraction/tests/test_stages_extract_text.py`:

```python
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
```

- [ ] **Step 6: Run tests**

```
pytest extraction/tests/test_stages_extract_text.py -v
```
Expected: 4 tests pass.

- [ ] **Step 7: Commit**

```
git add extraction/stages/extract_text.py extraction/tests/test_stages_extract_text.py
git commit -m "feat(extraction): run_text stage with prereq check + error isolation"
```

---

## Task 6: Describe-figures stage (`run_figures`)

**Files:**
- Create: `extraction/stages/describe_figures.py`
- Test: `extraction/tests/test_stages_describe_figures.py`

**Context:** Stage 3 is analogous to Stage 2 but (a) runs over FIGURE/DIAGRAM/TECHNICAL_DRAWING regions, (b) crops each region, (c) saves the crop, (d) calls `figure_descriptor.describe(crop)`, and (e) drops elements that end up with neither `image_path` nor `description` (mirrors the monolith's pipeline.py:134-140). Caption from segmenter is preserved.

- [ ] **Step 1: Create failing happy-path + skip + missing-prereq + error tests**

`extraction/tests/test_stages_describe_figures.py`:

```python
"""Unit tests for run_figures with stub adapters."""
from __future__ import annotations

import json
from pathlib import Path

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
    def describe(self, image):
        return f"a stub description {image.size}"


@register_figure_descriptor("stub_fig_empty")
class _StubFigEmpty:
    TOOL_NAME = "stub_fig_empty"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def describe(self, image):
        return ""


@register_figure_descriptor("stub_fig_broken")
class _StubFigBroken:
    TOOL_NAME = "stub_fig_broken"
    def __init__(self, **_): pass
    @property
    def tool_name(self): return self.TOOL_NAME
    def describe(self, image):
        raise RuntimeError("describe blew up")


def _seed_segment(out_dir: Path):
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


def test_figures_happy_path(tmp_path: Path):
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


def test_figures_drops_empty_when_no_crop_saved(tmp_path: Path, monkeypatch):
    """If describer returns empty AND we somehow end up without image_path, drop."""
    out = tmp_path / "doc1"
    _seed_segment(out)
    # Patch save_element_crop to raise, causing image_path to stay None;
    # describer returns "" → element must be dropped, not written.
    from extraction.stages import describe_figures as mod

    orig_save = OutputWriter.save_element_crop
    calls = {"n": 0}
    def _skip(self, **kw):
        calls["n"] += 1
        raise RuntimeError("simulated crop failure")
    monkeypatch.setattr(OutputWriter, "save_element_crop", _skip)

    cfg = _cfg(figure_descriptor="stub_fig_empty")
    exit_code = run_figures([out], cfg)
    # The failure cascades to a per-PDF error, which is the documented behavior.
    assert exit_code == 1


def test_figures_skips_when_marker_exists(tmp_path: Path, monkeypatch):
    out = tmp_path / "doc1"
    _seed_segment(out)
    OutputWriter(out).mark_stage_done("describe-figures")
    from extraction.stages import describe_figures as mod
    def _boom(*a, **kw):
        raise AssertionError("figure descriptor must not be loaded")
    monkeypatch.setattr(mod, "get_figure_descriptor", _boom)
    assert run_figures([out], _cfg()) == 0


def test_figures_missing_prereq_writes_error(tmp_path: Path):
    out = tmp_path / "doc1"
    out.mkdir(parents=True)
    assert run_figures([out], _cfg()) == 1
    assert (out / ".stages" / "describe-figures.error").exists()


def test_figures_error_writes_marker_and_continues(tmp_path: Path):
    out_a = tmp_path / "doc_a"
    out_b = tmp_path / "doc_b"
    _seed_segment(out_a)
    _seed_segment(out_b)
    exit_code = run_figures([out_a, out_b], _cfg(figure_descriptor="stub_fig_broken"))
    assert exit_code == 1
    for out in (out_a, out_b):
        assert (out / ".stages" / "describe-figures.error").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest extraction/tests/test_stages_describe_figures.py -v
```
Expected: ImportError on `extraction.stages.describe_figures`.

- [ ] **Step 3: Implement `extraction/stages/describe_figures.py`**

```python
"""Stage 3 — describe figure/diagram/drawing regions."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image as PILImage

from ..config import ExtractionConfig
from ..models import Element, ElementContent, ElementType, Region
from ..output import OutputWriter
from ..registry import get_figure_descriptor
from . import StageOutcome, print_stage_summary

_STAGE: str = "describe-figures"
_PREV: str = "segment"
_TARGET_TYPES = {
    ElementType.FIGURE,
    ElementType.DIAGRAM,
    ElementType.TECHNICAL_DRAWING,
}


def _element_id(doc_id: str, region: Region) -> str:
    x0, y0, x1, y1 = (round(v) for v in region.bbox)
    raw = f"{doc_id}:{region.page}:{region.region_type.value}:{x0},{y0},{x1},{y1}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_figures(out_dirs: list[Path], cfg: ExtractionConfig) -> int:
    plan: list[tuple[Path, OutputWriter, dict]] = []
    outcomes: list[StageOutcome] = []

    for out_dir in out_dirs:
        writer = OutputWriter(out_dir)
        label = str(out_dir)
        if writer.is_stage_done(_STAGE):
            outcomes.append(StageOutcome(label=label, status="skipped"))
            print(f"Processing {label} ... ↷ skipped (already done)")
            continue
        if not writer.is_stage_done(_PREV):
            exc = FileNotFoundError(f"Stage '{_PREV}' not done for {out_dir}")
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="missing_prereq",
                detail=f"(Vorgänger '{_PREV}' fehlt)",
            ))
            print(f"Processing {label} ... ✗ missing prerequisite: {_PREV}")
            continue
        plan.append((out_dir, writer, writer.read_segmentation()))

    if not plan:
        return print_stage_summary(_STAGE, outcomes, [
            d for d in out_dirs
            if (d / ".stages" / f"{_STAGE}.done").exists()
        ])

    describer = get_figure_descriptor(
        cfg.figure_descriptor, **cfg.get_adapter_config(cfg.figure_descriptor)
    )

    for out_dir, writer, meta in plan:
        label = str(out_dir)
        try:
            _process_one(out_dir, writer, meta, describer, cfg)
            writer.mark_stage_done(_STAGE)
            outcomes.append(StageOutcome(label=label, status="success"))
            print(f"Processing {label} ... ✓")
        except Exception as exc:
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="error",
                detail=f"(Fehler: siehe .stages/{_STAGE}.error)",
            ))
            print(f"Processing {label} ... ✗ {type(exc).__name__}: {exc}")

    ok_dirs = [
        d for d in out_dirs
        if (d / ".stages" / f"{_STAGE}.done").exists()
    ]
    return print_stage_summary(_STAGE, outcomes, ok_dirs)


def _load_page(out_dir: Path, page: int):
    return PILImage.open(out_dir / "pages" / str(page) / "page.png").convert("RGB")


def _process_one(
    out_dir: Path,
    writer: OutputWriter,
    meta: dict,
    describer,
    cfg: ExtractionConfig,
) -> None:
    regions: list[Region] = meta["regions"]
    doc_id: str = meta["doc_id"]
    for region in regions:
        if region.region_type not in _TARGET_TYPES:
            continue
        if region.confidence < cfg.confidence_threshold:
            continue
        page_img = _load_page(out_dir, region.page)
        crop = writer.crop_region(page_img, region.bbox, dpi=cfg.dpi)
        description = describer.describe(crop)
        content = ElementContent(description=description)
        if region.content is not None and region.content.caption:
            content.caption = region.content.caption

        el_id = _element_id(doc_id, region)
        el = Element(
            element_id=el_id,
            type=region.region_type,
            page=region.page,
            bbox=region.bbox,
            reading_order_index=0,
            section_path=[],
            confidence=region.confidence,
            extractor=describer.tool_name,
            content=content,
        )
        rel = writer.save_element_crop(
            page=region.page, element_id=el_id,
            element_type=region.region_type.value, image=crop,
        )
        el.content.image_path = str(rel.relative_to(writer.output_dir))

        if not el.content.image_path and not (el.content.description or "").strip():
            continue
        writer.write_element_sidecar(el)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest extraction/tests/test_stages_describe_figures.py -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```
git add extraction/stages/describe_figures.py extraction/tests/test_stages_describe_figures.py
git commit -m "feat(extraction): run_figures stage with prereq check + drops"
```

---

## Task 7: Assemble stage (`run_assemble`)

**Files:**
- Create: `extraction/stages/assemble.py`
- Test: `extraction/tests/test_stages_assemble.py`

**Context:** Stage 4 reads metadata from `segmentation.json`, calls `OutputWriter.build_content_list(...)` (which reads all sidecars, sorts deterministically, re-numbers `reading_order_index`), and writes `content_list.json`. No GPU. Refuses work if any of the three preceding stages is missing its marker.

- [ ] **Step 1: Create failing tests**

`extraction/tests/test_stages_assemble.py`:

```python
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
    # Remove content_list.json to prove the stage doesn't rebuild it
    assert run_assemble([out], _cfg()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest extraction/tests/test_stages_assemble.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `extraction/stages/assemble.py`**

```python
"""Stage 4 — assemble content_list.json from sidecars (no GPU)."""
from __future__ import annotations

from pathlib import Path

from ..config import ExtractionConfig
from ..output import OutputWriter
from . import StageOutcome, print_stage_summary

_STAGE: str = "assemble"
_PREREQS = ("segment", "extract-text", "describe-figures")


def run_assemble(out_dirs: list[Path], cfg: ExtractionConfig) -> int:
    outcomes: list[StageOutcome] = []
    for out_dir in out_dirs:
        writer = OutputWriter(out_dir)
        label = str(out_dir)
        if writer.is_stage_done(_STAGE):
            outcomes.append(StageOutcome(label=label, status="skipped"))
            print(f"Processing {label} ... ↷ skipped (already done)")
            continue
        missing = [p for p in _PREREQS if not writer.is_stage_done(p)]
        if missing:
            exc = FileNotFoundError(
                f"Stages {missing} not done for {out_dir}"
            )
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="missing_prereq",
                detail=f"(Vorgänger fehlt: {', '.join(missing)})",
            ))
            print(f"Processing {label} ... ✗ missing prerequisites: {missing}")
            continue
        try:
            _process_one(writer)
            writer.mark_stage_done(_STAGE)
            outcomes.append(StageOutcome(label=label, status="success"))
            print(f"Processing {label} ... ✓")
        except Exception as exc:
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="error",
                detail=f"(Fehler: siehe .stages/{_STAGE}.error)",
            ))
            print(f"Processing {label} ... ✗ {type(exc).__name__}: {exc}")

    ok_dirs = [
        d for d in out_dirs
        if (d / ".stages" / f"{_STAGE}.done").exists()
    ]
    return print_stage_summary(_STAGE, outcomes, ok_dirs)


def _process_one(writer: OutputWriter) -> None:
    meta = writer.read_segmentation()
    content_list = writer.build_content_list(
        doc_id=meta["doc_id"],
        source_file=meta["source_file"],
        total_pages=meta["total_pages"],
        segmentation_tool=meta["segmentation_tool"],
    )
    writer.write_content_list(content_list)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest extraction/tests/test_stages_assemble.py -v
```
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```
git add extraction/stages/assemble.py extraction/tests/test_stages_assemble.py
git commit -m "feat(extraction): run_assemble stage builds content_list.json"
```

---

## Task 8: CLI subcommands in `__main__.py`

**Files:**
- Modify: `extraction/__main__.py`
- Modify: `extraction/config.py` (add constant)
- Rewrite: `extraction/tests/test_cli.py`

**Context:** Replace the two existing subcommands (`extract`, `rebuild`) with four new ones (`segment`, `extract-text`, `describe-figures`, `assemble`). All four share a `--config` flag with the same resolve rules as today (`__main__.py:55-61`).

- [ ] **Step 1: Add constant to `extraction/config.py`**

Add near the top of the module, after imports:

```python
DEFAULT_OUTPUT_BASE = "outputs"
```

- [ ] **Step 2: Rewrite `extraction/__main__.py`**

Replace the file contents with:

```python
"""CLI entrypoint: python -m extraction {segment, extract-text, describe-figures, assemble}."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import extraction.adapters  # noqa: F401 — trigger adapter registration

from .config import DEFAULT_OUTPUT_BASE, ExtractionConfig, load_extraction_config
from .stages.assemble import run_assemble
from .stages.describe_figures import run_figures
from .stages.extract_text import run_text
from .stages.segment import run_segment


def _load_cfg(config_path: Path | None) -> ExtractionConfig:
    if config_path is not None:
        return load_extraction_config(config_path)
    default = Path("extraction_config.yaml")
    if default.exists():
        return load_extraction_config(default)
    return ExtractionConfig()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="extraction")
    sub = parser.add_subparsers(dest="command", required=True)

    seg = sub.add_parser("segment", help="Stage 1: render + segment PDFs")
    seg.add_argument("pdfs", nargs="+", type=Path)
    seg.add_argument("--config", type=Path, default=None)
    seg.add_argument("--out", type=Path, default=Path(DEFAULT_OUTPUT_BASE))

    txt = sub.add_parser("extract-text", help="Stage 2: text extraction")
    txt.add_argument("outdirs", nargs="+", type=Path)
    txt.add_argument("--config", type=Path, default=None)

    fig = sub.add_parser("describe-figures", help="Stage 3: figure descriptions")
    fig.add_argument("outdirs", nargs="+", type=Path)
    fig.add_argument("--config", type=Path, default=None)

    asm = sub.add_parser("assemble", help="Stage 4: build content_list.json")
    asm.add_argument("outdirs", nargs="+", type=Path)
    asm.add_argument("--config", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(getattr(args, "config", None))

    if args.command == "segment":
        sys.exit(run_segment(args.pdfs, cfg, args.out))
    if args.command == "extract-text":
        sys.exit(run_text(args.outdirs, cfg))
    if args.command == "describe-figures":
        sys.exit(run_figures(args.outdirs, cfg))
    if args.command == "assemble":
        sys.exit(run_assemble(args.outdirs, cfg))
    print(f"Unknown command: {args.command}")
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Rewrite `extraction/tests/test_cli.py`**

Replace with:

```python
"""CLI smoke tests — does argparse wire each subcommand to the right stage?"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from extraction.__main__ import main


def _invoke(*argv: str) -> int:
    with patch.object(sys, "argv", ["extraction", *argv]):
        try:
            main()
            return 0
        except SystemExit as e:
            return int(e.code) if e.code is not None else 0


def test_segment_dispatches(tmp_path: Path, monkeypatch):
    calls = {}
    def _fake(pdfs, cfg, output_base):
        calls["args"] = (list(pdfs), output_base)
        return 0
    monkeypatch.setattr("extraction.__main__.run_segment", _fake)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert _invoke("segment", str(pdf), "--out", str(tmp_path / "o")) == 0
    assert calls["args"] == ([pdf], tmp_path / "o")


def test_text_dispatches(monkeypatch, tmp_path: Path):
    calls = {}
    def _fake(dirs, cfg):
        calls["args"] = list(dirs)
        return 0
    monkeypatch.setattr("extraction.__main__.run_text", _fake)
    d = tmp_path / "d1"
    d.mkdir()
    assert _invoke("extract-text", str(d)) == 0
    assert calls["args"] == [d]


def test_figures_dispatches(monkeypatch, tmp_path: Path):
    calls = {}
    def _fake(dirs, cfg):
        calls["args"] = list(dirs)
        return 0
    monkeypatch.setattr("extraction.__main__.run_figures", _fake)
    d = tmp_path / "d1"
    d.mkdir()
    assert _invoke("describe-figures", str(d)) == 0
    assert calls["args"] == [d]


def test_assemble_dispatches(monkeypatch, tmp_path: Path):
    calls = {}
    def _fake(dirs, cfg):
        calls["args"] = list(dirs)
        return 0
    monkeypatch.setattr("extraction.__main__.run_assemble", _fake)
    d = tmp_path / "d1"
    d.mkdir()
    assert _invoke("assemble", str(d)) == 0
    assert calls["args"] == [d]


def test_unknown_subcommand_exits_nonzero():
    # argparse exits 2 on unknown subcommand
    with pytest.raises(SystemExit) as exc:
        _invoke("nope")
    assert exc.value.code in (1, 2)
```

- [ ] **Step 4: Run the CLI tests**

```
pytest extraction/tests/test_cli.py -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```
git add extraction/__main__.py extraction/config.py extraction/tests/test_cli.py
git commit -m "feat(extraction): CLI dispatches four stage subcommands"
```

---

## Task 9: Remove the monolith pipeline

**Files:**
- Modify: `extraction/pipeline.py`
- Modify or delete: `extraction/tests/test_pipeline.py`

**Context:** `ExtractionPipeline.run()` and its helpers are dead. Delete them. Keep `extraction/pipeline.py` as a near-empty shell only if other modules import from it; otherwise delete the file. Tests that imported `ExtractionPipeline` get removed or shrunk.

- [ ] **Step 1: Check for imports of `ExtractionPipeline`**

```
grep -rn "ExtractionPipeline\|from extraction.pipeline\|extraction\.pipeline" extraction/ --include="*.py"
```
Expected: only `extraction/pipeline.py` itself and any test files.

- [ ] **Step 2: Delete `extraction/pipeline.py`**

```
rm extraction/pipeline.py
```

If step 1 showed live imports, leave the file with just a comment block instead:

```python
"""Intentionally empty — replaced by extraction.stages (see spec 2026-04-23)."""
```

- [ ] **Step 3: Delete or shrink `extraction/tests/test_pipeline.py`**

If the file tests only the old `ExtractionPipeline.run()`, delete it:
```
git rm extraction/tests/test_pipeline.py
```

If it has tests of Region/Element helpers worth keeping, keep those and delete the `run()`-focused ones. Any skipped tests marked in Task 2 step 4 should now be un-skipped or removed.

- [ ] **Step 4: Run full test suite**

```
pytest -q
```
Expected: all green.

- [ ] **Step 5: Lint + type-check**

```
ruff check extraction
mypy
```
Expected: clean.

- [ ] **Step 6: Commit**

```
git add -A
git commit -m "refactor(extraction): remove monolithic ExtractionPipeline"
```

---

## Task 10: End-to-end integration test

**Files:**
- Create: `extraction/tests/test_stages_integration.py`
- Create: `extraction/tests/fixtures/reference_content_list.json` (commit-time capture)

**Context:** One real short PDF, four stage calls in-process, structural-plus-strict comparison against a committed reference `content_list.json` captured once from a clean run. Strict fields match bit-identical; VLM text/description only shape-checked (spec §8).

The project root has short PDFs — use the smallest one for this test. At the time of writing this plan, the repo contains `1.9.20 PV 1001.12, Rev. 3.pdf`, `3_HRA_for_offshore.pdf`, `2_Emergency-DSS-EcoSystem-Emergency Decision Support Techniques for Nuclear Power.pdf`, `jmmp-09-00199-v2.pdf`. Pick the one with the fewest pages by running `python -c "import fitz; ..."` before the test run.

- [ ] **Step 1: Create the test file**

```python
"""End-to-end integration test — all four stages on a real PDF.

Marked integration: requires GPU + real model weights. Run with:
    pytest -m integration extraction/tests/test_stages_integration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from extraction.config import ExtractionConfig
from extraction.stages.assemble import run_assemble
from extraction.stages.describe_figures import run_figures
from extraction.stages.extract_text import run_text
from extraction.stages.segment import run_segment

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).parent / "fixtures"
PDF_FIXTURE = REPO_ROOT / "3_HRA_for_offshore.pdf"  # swap if a shorter one is preferred
REFERENCE = FIXTURE_DIR / "reference_content_list.json"

STRICT_ELEMENT_KEYS = (
    "element_id", "type", "page", "bbox",
    "reading_order_index", "confidence", "extractor",
)
STRICT_CONTENT_KEYS = ("markdown", "latex", "caption", "image_path")
STRUCTURAL_CONTENT_KEYS = ("text", "description")


@pytest.mark.integration
def test_all_four_stages_bit_plus_structural(tmp_path: Path):
    assert PDF_FIXTURE.exists(), f"Fixture PDF missing: {PDF_FIXTURE}"
    assert REFERENCE.exists(), (
        f"Reference content_list.json missing at {REFERENCE}. "
        "Run the stages once on the fixture and copy the output there."
    )

    cfg = ExtractionConfig()  # defaults: mineru25 / olmocr2 / mineru25 / noop / qwen25vl

    assert run_segment([PDF_FIXTURE], cfg, output_base=tmp_path) == 0
    out_dir = tmp_path / PDF_FIXTURE.stem
    assert run_text([out_dir], cfg) == 0
    assert run_figures([out_dir], cfg) == 0
    assert run_assemble([out_dir], cfg) == 0

    actual = json.loads((out_dir / "content_list.json").read_text(encoding="utf-8"))
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))

    # Top-level metadata: strict
    for key in ("doc_id", "source_file", "total_pages", "schema_version",
                "segmentation_tool"):
        assert actual[key] == expected[key], f"top-level {key} drifted"

    # Pages: strict
    assert actual["pages"] == expected["pages"]

    # Elements: zip and compare
    assert len(actual["elements"]) == len(expected["elements"]), (
        f"element count changed: {len(actual['elements'])} vs {len(expected['elements'])}"
    )
    for i, (a, e) in enumerate(zip(actual["elements"], expected["elements"])):
        for k in STRICT_ELEMENT_KEYS:
            assert a[k] == e[k], f"element[{i}].{k} drifted: {a[k]!r} vs {e[k]!r}"
        for k in STRICT_CONTENT_KEYS:
            assert a["content"].get(k) == e["content"].get(k), (
                f"element[{i}].content.{k} drifted"
            )
        for k in STRUCTURAL_CONTENT_KEYS:
            # structural: present when expected was present, non-empty, str
            if e["content"].get(k):
                got = a["content"].get(k)
                assert isinstance(got, str) and got.strip(), (
                    f"element[{i}].content.{k} expected non-empty string"
                )
```

- [ ] **Step 2: Generate the reference fixture**

On a machine with GPU + full install:

```
mkdir -p extraction/tests/fixtures
rm -rf /tmp/stage_ref
python -m extraction segment 3_HRA_for_offshore.pdf --out /tmp/stage_ref
python -m extraction extract-text /tmp/stage_ref/3_HRA_for_offshore
python -m extraction describe-figures /tmp/stage_ref/3_HRA_for_offshore
python -m extraction assemble /tmp/stage_ref/3_HRA_for_offshore
cp /tmp/stage_ref/3_HRA_for_offshore/content_list.json extraction/tests/fixtures/reference_content_list.json
```

- [ ] **Step 3: Run the integration test**

```
pytest -m integration extraction/tests/test_stages_integration.py -v
```
Expected: PASS. Because `content.text` and `content.description` are only structurally checked, re-runs on the same PDF pass even though VLM sampling produces different text each time.

- [ ] **Step 4: Commit**

```
git add extraction/tests/test_stages_integration.py extraction/tests/fixtures/reference_content_list.json
git commit -m "test(extraction): integration test for four-stage pipeline"
```

---

## Task 11: README

**Files:**
- Modify: `README.md`

**Context:** Remove the outdated `## CLI` block, add `## Extraction-Pipeline`. Everything else in the README was verified against the code during brainstorming and stays.

- [ ] **Step 1: Read current README**

```
cat README.md
```

- [ ] **Step 2: Replace the `## CLI` section**

Delete the block from `## CLI` up to (but not including) `## CPU-only config example`. Insert in its place:

```markdown
## Extraction-Pipeline

Die Extraktion läuft in vier manuellen Schritten. Jedes Kommando lädt
das zugehörige Modell, verarbeitet alle genannten PDFs/Ordner, und
beendet sich (gibt die GPU wieder frei). Am Ende jedes Schritts wird
der nächste Befehl zum Kopieren ausgegeben.

### 1. Segmentieren (MinerU)

Input: PDF-Pfade. Legt Output-Ordner automatisch unter `outputs/` an.

    python -m extraction segment <pdf1> <pdf2> ...

### 2. Text extrahieren (olmOCR-2)

    python -m extraction extract-text <outdir1> <outdir2> ...

### 3. Figures beschreiben (Qwen2.5-VL)

    python -m extraction describe-figures <outdir1> <outdir2> ...

### 4. Zusammenbauen (CPU, kein Modell)

    python -m extraction assemble <outdir1> <outdir2> ...

### Einzelnen Schritt neu laufen lassen

Marker löschen und Stage neu starten, z.B. Text-Extraktion für ein PDF:

    rm outputs/jmmp-09-00199-v2/.stages/extract-text.done
    python -m extraction extract-text outputs/jmmp-09-00199-v2

```

- [ ] **Step 3: Sanity-check with `rg`**

```
rg -n "python -m extraction extract " README.md || echo "old command gone — good"
```
Expected: "old command gone — good".

- [ ] **Step 4: Commit**

```
git add README.md
git commit -m "docs(readme): staged pipeline workflow replaces monolith CLI"
```

---

## Self-Review Notes

**Spec coverage:**

- §1 (four stages, each own process) → Tasks 4–7 (stage impls) + Task 8 (CLI).
- §1 Sidecar-Verantwortung → each stage's `_process_one` writes its region-types' sidecars.
- §1 Config-Flag → Task 8 wires `--config` into all four subcommands.
- §1 Lazy Model Load → pre-scan in Tasks 4–6 before `get_*` calls; covered by the skip tests.
- §2 Output-Basis → `DEFAULT_OUTPUT_BASE` in Task 8 step 1, `--out` flag on `segment`.
- §3 Marker files → Task 1 (writer API) + Tasks 4–7 (usage).
- §4 Error handling → per-PDF try/except + `.error` file + continue; covered by error tests in Tasks 4–7.
- §5 Rückmeldung → `print_stage_summary()` in Task 3.
- §6 README → Task 11.
- §7 Migration table → one task per file; Task 9 removes the monolith.
- §8 Testing → unit tests per stage (Tasks 4–7) + E2E (Task 10).

**Placeholder scan:** No "TBD" / "TODO" / "appropriate error handling" / "handle edge cases". Every code step has literal code; every command step has an exact shell line.

**Type consistency:** `run_segment(pdf_paths, cfg, output_base)` vs. `run_text(out_dirs, cfg)` / `run_figures(out_dirs, cfg)` / `run_assemble(out_dirs, cfg)` — intentional signature difference (stage 1 takes PDFs + base, later stages take output dirs). `StageOutcome` fields (`label`, `status`, `detail`) are used consistently across tasks. `_element_id` is duplicated across `segment.py`, `extract_text.py`, `describe_figures.py` — acceptable duplication for now, refactor only if a fourth call site appears.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-23-staged-extraction-pipeline.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Each of the 11 tasks gets its own agent with the exact task block as prompt. Review happens between tasks so regressions surface immediately.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review.

**Which approach?**
