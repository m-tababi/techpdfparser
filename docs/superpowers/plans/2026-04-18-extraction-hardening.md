# Extraction Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix coordinate handling, element IDs, merge-rule, packaging, output isolation, confidence filter, and empty-content handling in the extraction block before the next iteration.

**Architecture:** The fixes touch `extraction/pipeline.py`, `extraction/output.py`, `extraction/adapters/mineru25_segmenter.py`, `extraction/config.py` (wiring), and `pyproject.toml`. The new merge rule is "Config dictates, Pipeline obeys" — layout from segmenter, content from configured role tool, tool-match optimization only when role tool == segmenter tool.

**Tech Stack:** Python 3.10+, Pydantic 2.x, PyMuPDF, MinerU 2.5, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-04-18-extraction-hardening-design.md`

---

## File Structure

Files that will be created or modified:

- `pyproject.toml` — add package discovery + `pymupdf` runtime + `[gpu]` extra
- `extraction/pipeline.py` — merge rewrite, fail-safe, bbox-based IDs, filename-only source, dpi param, empty-content drop, cleanups
- `extraction/output.py` — scale + clamp in `crop_region`
- `extraction/adapters/mineru25_segmenter.py` — read real confidence from `layout_dets[*].score`
- `extraction/__main__.py` — wire `cfg.dpi` into adapter config when not overridden
- `extraction/tests/test_pipeline.py` — tests for fail-safe, merge-rule, empty-content, IDs, dpi plumb-through
- `extraction/tests/test_output.py` — tests for crop scaling + clamping
- `extraction/tests/test_mineru25_segmenter.py` (new) — confidence-mapping test with a small fixture
- `extraction/tests/test_cli.py` — update for new source_file behavior
- `docs/extraction_output.md` — coordinate system, merge rules, IDs, Phase 2 note
- `docs/writing_adapters.md` (new) — adapter author checklist
- `README.md` — install commands, CPU config example, commands
- `backlog.md` — two English entries
- `tasks/extraction_completion_plan.md`, `tasks/todo.md` — add ARCHIVED markers

---

## Task 1: Packaging Fix (pyproject.toml)

**Files:**
- Modify: `pyproject.toml`

**Context:** `pip install -e .` currently fails because setuptools tries to auto-discover packages in `outputs/`, `_archive/`, etc. `pymupdf` is the default renderer but not a declared runtime dep.

- [ ] **Step 1: Write a smoke-test commit that proves the failure**

Run: `python -m pip install -e . --dry-run`
Expected: error about multiple top-level packages (setuptools discovery conflict), or missing `fitz` at runtime.

- [ ] **Step 2: Update pyproject.toml**

Replace the file contents with:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "techpdfparser"
version = "0.1.0"
description = "A technical PDF parsing library"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "pillow>=10.0",
    "pymupdf>=1.23",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
    "mypy",
]
gpu = [
    "mineru>=2.5",
    "transformers>=4.40",
    "torch>=2.2",
    "beautifulsoup4>=4.12",
]

[tool.setuptools.packages.find]
include = ["extraction*"]
exclude = ["outputs*", "_archive*", "embedding*", "indexing*", "tests*", "tasks*"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
files = [
    "extraction",
]
check_untyped_defs = true
disallow_incomplete_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unreachable = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = ["yaml"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["mineru.*", "transformers.*", "torch.*", "bs4.*", "fitz.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["extraction/tests"]
addopts = "-m 'not integration'"
markers = [
    "integration: heavy end-to-end tests that require MinerU / GPU / extra models",
]
```

- [ ] **Step 3: Verify install works**

Run: `venv/bin/pip install -e .`
Expected: successful install, no "multiple top-level packages" error.

- [ ] **Step 4: Verify tests still pass**

Run: `venv/bin/pytest -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore(packaging): pin pymupdf runtime dep and explicit package discovery"
```

---

## Task 2: source_file uses only the filename

**Files:**
- Modify: `extraction/pipeline.py:132`
- Test: `extraction/tests/test_pipeline.py`

**Context:** `content_list.json:source_file` currently holds the full absolute path. Spec Section 3 says filename only.

- [ ] **Step 1: Write failing test**

Append to `extraction/tests/test_pipeline.py`:

```python
def test_pipeline_source_file_is_filename_only(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "dir"
    nested.mkdir(parents=True)
    pdf_path = nested / "report.pdf"
    pdf_path.write_bytes(b"fake")
    output_dir = tmp_path / "output"

    pipeline = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=output_dir,
        confidence_threshold=0.3,
    )
    pipeline.run(pdf_path)

    data = json.loads((output_dir / "content_list.json").read_text())
    assert data["source_file"] == "report.pdf"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py::test_pipeline_source_file_is_filename_only -v`
Expected: FAIL (current behavior returns full path).

- [ ] **Step 3: Apply the one-line fix**

In `extraction/pipeline.py`, change line 132 from:

```python
            source_file=str(pdf_path),
```

to:

```python
            source_file=pdf_path.name,
```

- [ ] **Step 4: Verify test passes**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py::test_pipeline_source_file_is_filename_only -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `venv/bin/pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add extraction/pipeline.py extraction/tests/test_pipeline.py
git commit -m "fix(extraction): store filename only in content_list.source_file"
```

---

## Task 3: Pipeline knows DPI; config wires top-level dpi into adapter config

**Files:**
- Modify: `extraction/pipeline.py` (add `dpi` init param)
- Modify: `extraction/__main__.py` (wire `cfg.dpi` to renderer adapter config when not overridden)
- Test: `extraction/tests/test_config.py`
- Test: `extraction/tests/test_pipeline.py`

**Context:** Spec Section 9: top-level `dpi` must reach the renderer. The pipeline also needs `dpi` for crop scaling in Task 4.

- [ ] **Step 1: Read current test_config.py head**

Run: `venv/bin/pytest extraction/tests/test_config.py -q`
Expected: all current tests pass. Confirm baseline.

- [ ] **Step 2: Write failing test for `_load_cfg` / adapter wiring**

Append to `extraction/tests/test_config.py`:

```python
def test_top_level_dpi_flows_to_renderer_when_not_overridden(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("extraction:\n  dpi: 300\n", encoding="utf-8")
    from extraction.config import load_extraction_config
    cfg = load_extraction_config(cfg_path)
    resolved = _resolve_renderer_dpi(cfg)
    assert resolved == 300


def test_adapter_dpi_overrides_top_level_dpi(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "extraction:\n  dpi: 300\nadapters:\n  pymupdf:\n    dpi: 450\n",
        encoding="utf-8",
    )
    from extraction.config import load_extraction_config
    cfg = load_extraction_config(cfg_path)
    resolved = _resolve_renderer_dpi(cfg)
    assert resolved == 450
```

Add this helper at the top of the test file (imported from the CLI module):

```python
from extraction.__main__ import _resolve_renderer_dpi
```

- [ ] **Step 3: Run — expect ImportError**

Run: `venv/bin/pytest extraction/tests/test_config.py -v`
Expected: ImportError — `_resolve_renderer_dpi` does not exist yet.

- [ ] **Step 4: Implement `_resolve_renderer_dpi` in `extraction/__main__.py`**

Add near `_load_cfg`:

```python
def _resolve_renderer_dpi(cfg: ExtractionConfig) -> int:
    """Top-level cfg.dpi unless the renderer adapter block overrides it."""
    adapter_cfg = cfg.get_adapter_config(cfg.renderer)
    if "dpi" in adapter_cfg:
        return int(adapter_cfg["dpi"])
    return int(cfg.dpi)
```

Then modify `_run_extract` to use it. Replace:

```python
    renderer = get_renderer(cfg.renderer, **cfg.get_adapter_config(cfg.renderer))
```

with:

```python
    renderer_kwargs = dict(cfg.get_adapter_config(cfg.renderer))
    renderer_kwargs.setdefault("dpi", cfg.dpi)
    renderer = get_renderer(cfg.renderer, **renderer_kwargs)
    dpi = _resolve_renderer_dpi(cfg)
```

And add `dpi=dpi` to the `ExtractionPipeline(...)` call.

- [ ] **Step 5: Add `dpi` to `ExtractionPipeline.__init__`**

In `extraction/pipeline.py`, change the signature and store it:

```python
    def __init__(
        self,
        renderer: PageRenderer,
        segmenter: Segmenter,
        text_extractor: TextExtractor,
        table_extractor: TableExtractor,
        formula_extractor: FormulaExtractor,
        figure_descriptor: FigureDescriptor,
        output_dir: Path,
        confidence_threshold: float = 0.3,
        dpi: int = 150,
    ) -> None:
        self.renderer = renderer
        self.segmenter = segmenter
        self.text_extractor = text_extractor
        self.table_extractor = table_extractor
        self.formula_extractor = formula_extractor
        self.figure_descriptor = figure_descriptor
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.dpi = dpi
```

- [ ] **Step 6: Run tests — expect green**

Run: `venv/bin/pytest extraction/tests/test_config.py extraction/tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 7: Full suite**

Run: `venv/bin/pytest -q`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add extraction/pipeline.py extraction/__main__.py extraction/tests/test_config.py
git commit -m "feat(extraction): wire top-level dpi to renderer and pipeline"
```

---

## Task 4: Crop scaling and clamping in OutputWriter.crop_region

**Files:**
- Modify: `extraction/output.py:68-70`
- Modify: `extraction/pipeline.py` (pass `dpi` through to crop_region)
- Test: `extraction/tests/test_output.py`

**Context:** Spec Section 1: bbox stays in PDF-points. Scale `dpi/72` is applied only at crop time. Negative/overflow coords get clamped to image bounds.

- [ ] **Step 1: Write failing tests**

Append to `extraction/tests/test_output.py`:

```python
def test_crop_region_scales_points_to_pixels(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    page_img = Image.new("RGB", (1240, 1754), color="white")  # ~A4 at 150 DPI
    bbox_points = [100.0, 200.0, 200.0, 300.0]
    crop = writer.crop_region(page_img, bbox_points, dpi=150)
    # 100 pt * 150/72 = 208.33 → ceil-ish; 200 pt * 150/72 = 416.67 → 417
    # Resulting width = ceil(200*150/72) - floor(100*150/72) ≈ 209
    # Resulting height = ceil(300*150/72) - floor(200*150/72) ≈ 208
    assert crop.size == (209, 208)


def test_crop_region_clamps_to_image_bounds(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    page_img = Image.new("RGB", (500, 500), color="white")
    # bbox extends beyond image in every direction (in pixel space already)
    bbox_points = [-20.0, -20.0, 10_000.0, 10_000.0]
    crop = writer.crop_region(page_img, bbox_points, dpi=72)  # scale=1
    assert crop.size == (500, 500)


def test_crop_region_default_dpi_is_72_backwards_compat(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    page_img = Image.new("RGB", (1000, 800), color="white")
    # No dpi → default 72 → bbox treated as pixels
    crop = writer.crop_region(page_img, [100.0, 200.0, 500.0, 400.0], dpi=72)
    assert crop.size == (400, 200)
```

Update the existing `test_crop_from_page_image` to pass `dpi=72` explicitly:

```python
def test_crop_from_page_image(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    page_img = Image.new("RGB", (1000, 800), color="white")
    bbox = [100.0, 200.0, 500.0, 400.0]
    crop = writer.crop_region(page_img, bbox, dpi=72)

    assert crop.size == (400, 200)
```

- [ ] **Step 2: Run — expect signature mismatch**

Run: `venv/bin/pytest extraction/tests/test_output.py -v`
Expected: FAIL — `crop_region()` got unexpected keyword argument 'dpi'.

- [ ] **Step 3: Implement scale + clamp in `OutputWriter.crop_region`**

In `extraction/output.py`, replace:

```python
    def crop_region(self, page_image: Image, bbox: list[float]) -> Image:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        return page_image.crop((x0, y0, x1, y1))
```

with:

```python
    def crop_region(
        self, page_image: Image, bbox: list[float], dpi: int = 72
    ) -> Image:
        """Crop a region given in PDF-points from a page image rendered at `dpi`."""
        scale = dpi / 72.0
        x0 = max(0, int(bbox[0] * scale))
        y0 = max(0, int(bbox[1] * scale))
        x1 = min(page_image.width, int(bbox[2] * scale + 0.999))
        y1 = min(page_image.height, int(bbox[3] * scale + 0.999))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, page_image.width, page_image.height
        return page_image.crop((x0, y0, x1, y1))
```

- [ ] **Step 4: Update pipeline call sites**

In `extraction/pipeline.py`, update the three crop sites and the visual-save site to pass `self.dpi`:

```python
        elif region.region_type == ElementType.TABLE:
            crop = OutputWriter(self.output_dir).crop_region(
                page_img, region.bbox, dpi=self.dpi
            )
            return self.table_extractor.extract(crop, region.page)
        elif region.region_type == ElementType.FORMULA:
            crop = OutputWriter(self.output_dir).crop_region(
                page_img, region.bbox, dpi=self.dpi
            )
            return self.formula_extractor.extract(crop, region.page)
        elif region.region_type in {
            ElementType.FIGURE,
            ElementType.DIAGRAM,
            ElementType.TECHNICAL_DRAWING,
        }:
            crop = OutputWriter(self.output_dir).crop_region(
                page_img, region.bbox, dpi=self.dpi
            )
            description = self.figure_descriptor.describe(crop)
            return ElementContent(description=description)
```

And in the visual-save loop in `run()`:

```python
                crop = writer.crop_region(page_images[el.page], el.bbox, dpi=self.dpi)
```

- [ ] **Step 5: Run tests**

Run: `venv/bin/pytest extraction/tests/test_output.py extraction/tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 6: Full suite**

Run: `venv/bin/pytest -q && venv/bin/ruff check extraction && venv/bin/mypy`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add extraction/output.py extraction/pipeline.py extraction/tests/test_output.py
git commit -m "fix(extraction): scale PDF-point bboxes to pixels and clamp at crop time"
```

---

## Task 5: Bbox-based element_id

**Files:**
- Modify: `extraction/pipeline.py:88, 189-191`
- Test: `extraction/tests/test_pipeline.py`

**Context:** Spec Section 2: element_id = `sha256(doc_id:page:region_type:round(x0),round(y0),round(x1),round(y1))[:16]`. Pfadunabhängig, stabil gegen Reading-Order-Nondeterminismus.

- [ ] **Step 1: Write failing tests**

Append to `extraction/tests/test_pipeline.py`:

```python
def test_element_id_is_path_independent(tmp_path: Path) -> None:
    pdf_a = tmp_path / "a" / "doc.pdf"
    pdf_b = tmp_path / "b" / "doc.pdf"
    pdf_a.parent.mkdir(parents=True)
    pdf_b.parent.mkdir(parents=True)
    pdf_a.write_bytes(b"identical content")
    pdf_b.write_bytes(b"identical content")

    def run_one(pdf: Path, out: Path) -> list[str]:
        ExtractionPipeline(
            renderer=MockRenderer(),
            segmenter=MockSegmenter(),
            text_extractor=MockTextExtractor(),
            table_extractor=MockTableExtractor(),
            formula_extractor=MockFormulaExtractor(),
            figure_descriptor=MockFigureDescriptor(),
            output_dir=out,
            confidence_threshold=0.3,
        ).run(pdf)
        data = json.loads((out / "content_list.json").read_text())
        return [e["element_id"] for e in data["elements"]]

    ids_a = run_one(pdf_a, tmp_path / "out_a")
    ids_b = run_one(pdf_b, tmp_path / "out_b")
    assert ids_a == ids_b


def test_element_id_differs_on_region_type_at_same_bbox(tmp_path: Path) -> None:
    from extraction.pipeline import ExtractionPipeline as _P
    from extraction.models import Region
    pipe = _P(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=tmp_path,
        confidence_threshold=0.3,
    )
    r_text = Region(page=0, bbox=[10, 20, 30, 40], region_type=ElementType.TEXT, confidence=1.0)
    r_head = Region(page=0, bbox=[10, 20, 30, 40], region_type=ElementType.HEADING, confidence=1.0)
    id1 = pipe._make_element_id("docid", r_text)
    id2 = pipe._make_element_id("docid", r_head)
    assert id1 != id2
```

- [ ] **Step 2: Run — expect FAIL**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py::test_element_id_is_path_independent extraction/tests/test_pipeline.py::test_element_id_differs_on_region_type_at_same_bbox -v`
Expected: FAIL — current impl takes `pdf_path, region, seq` and hashes the path.

- [ ] **Step 3: Replace `_make_element_id`**

In `extraction/pipeline.py`, replace:

```python
    def _make_element_id(self, pdf_path: Path, region: Region, seq: int) -> str:
        raw = f"{pdf_path}:{region.page}:{region.region_type.value}:{seq}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

with:

```python
    def _make_element_id(self, doc_id: str, region: Region) -> str:
        x0, y0, x1, y1 = (round(v) for v in region.bbox)
        raw = (
            f"{doc_id}:{region.page}:{region.region_type.value}"
            f":{x0},{y0},{x1},{y1}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Update call site in `run`**

In `run()`, compute `doc_id` before the element loop and pass it in. Replace the snippet starting at line 82:

```python
        # 3. Route and extract
        elements: list[Element] = []
        for idx, region in enumerate(regions):
            content = self._extract_region(region, page_images)
            if content is None:
                continue

            element_id = self._make_element_id(pdf_path, region, idx)
```

with:

```python
        doc_id = self._make_doc_id(pdf_path)

        # 3. Route and extract
        elements: list[Element] = []
        for idx, region in enumerate(regions):
            content = self._extract_region(region, page_images)
            if content is None:
                continue

            element_id = self._make_element_id(doc_id, region)
```

And update the later `build_content_list` call so it uses the same `doc_id`:

```python
        content_list = writer.build_content_list(
            doc_id=doc_id,
            source_file=pdf_path.name,
            total_pages=page_count,
            segmentation_tool=self.segmenter.tool_name,
        )
```

- [ ] **Step 5: Run tests**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 6: Full suite + type checks**

Run: `venv/bin/pytest -q && venv/bin/ruff check extraction && venv/bin/mypy`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add extraction/pipeline.py extraction/tests/test_pipeline.py
git commit -m "fix(extraction): bbox-based element_id, path-independent"
```

---

## Task 6: Output-isolation fail-safe in Pipeline.run()

**Files:**
- Modify: `extraction/pipeline.py` (add fail-safe check at top of `run`)
- Test: `extraction/tests/test_pipeline.py`

**Context:** Spec Section 5: Pipeline aborts if output dir already holds `content_list.json`, `segmentation.json`, or non-empty `pages/`. No auto-cleanup, no `--overwrite`.

- [ ] **Step 1: Write failing tests**

Append to `extraction/tests/test_pipeline.py`:

```python
import pytest


def _make_pipeline(output_dir: Path) -> ExtractionPipeline:
    return ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=output_dir,
        confidence_threshold=0.3,
    )


def test_pipeline_aborts_when_content_list_exists(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()
    (out / "content_list.json").write_text("{}")

    with pytest.raises(FileExistsError):
        _make_pipeline(out).run(pdf_path)


def test_pipeline_aborts_when_segmentation_json_exists(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()
    (out / "segmentation.json").write_text("[]")

    with pytest.raises(FileExistsError):
        _make_pipeline(out).run(pdf_path)


def test_pipeline_aborts_when_pages_dir_is_nonempty(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    (out / "pages" / "0").mkdir(parents=True)
    (out / "pages" / "0" / "leftover.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(FileExistsError):
        _make_pipeline(out).run(pdf_path)


def test_pipeline_allows_empty_output_dir(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()
    _make_pipeline(out).run(pdf_path)  # must not raise
    assert (out / "content_list.json").exists()
```

- [ ] **Step 2: Run — expect FAIL**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py::test_pipeline_aborts_when_content_list_exists extraction/tests/test_pipeline.py::test_pipeline_aborts_when_segmentation_json_exists extraction/tests/test_pipeline.py::test_pipeline_aborts_when_pages_dir_is_nonempty -v`
Expected: FAIL — no guard present.

- [ ] **Step 3: Implement the check**

In `extraction/pipeline.py`, add a private method and call it at the top of `run()`:

```python
    def run(self, pdf_path: Path) -> ContentList:
        """Run the full extraction pipeline on a single PDF."""
        self._assert_output_dir_clean()
        writer = OutputWriter(self.output_dir)
        ...
```

And add the method near the other private helpers:

```python
    def _assert_output_dir_clean(self) -> None:
        """Refuse to mix artefacts from different runs in the same directory."""
        content_list = self.output_dir / "content_list.json"
        segmentation = self.output_dir / "segmentation.json"
        pages_dir = self.output_dir / "pages"
        conflicts: list[str] = []
        if content_list.exists():
            conflicts.append(str(content_list))
        if segmentation.exists():
            conflicts.append(str(segmentation))
        if pages_dir.exists() and any(pages_dir.iterdir()):
            conflicts.append(str(pages_dir))
        if conflicts:
            raise FileExistsError(
                "Extraction output dir already contains artefacts: "
                + ", ".join(conflicts)
                + ". Choose a different --output directory or remove these files."
            )
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + checks**

Run: `venv/bin/pytest -q && venv/bin/ruff check extraction && venv/bin/mypy`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add extraction/pipeline.py extraction/tests/test_pipeline.py
git commit -m "feat(extraction): fail-safe on non-empty output dir in Pipeline.run()"
```

---

## Task 7: Drop empty-content elements

**Files:**
- Modify: `extraction/pipeline.py` (add `_is_droppable` filter)
- Test: `extraction/tests/test_pipeline.py`

**Context:** Spec Section 7: text/heading with empty/whitespace text → drop. Table without markdown/text → drop. Formula without latex/text → drop. Visual element with only `image_path` → keep.

- [ ] **Step 1: Write failing tests**

Append to `extraction/tests/test_pipeline.py`:

```python
class EmptyTextSegmenter:
    """Emits text regions with empty content and a figure with only bbox."""
    tool_name = "empty_seg"

    def segment(self, pdf_path: Path) -> list[Region]:
        return [
            Region(
                page=0, bbox=[0, 0, 100, 50],
                region_type=ElementType.TEXT, confidence=0.9,
                content=ElementContent(text="   "),
            ),
            Region(
                page=0, bbox=[0, 60, 100, 200],
                region_type=ElementType.TEXT, confidence=0.9,
                content=ElementContent(text=""),
            ),
            Region(
                page=0, bbox=[0, 220, 100, 300],
                region_type=ElementType.FIGURE, confidence=0.9,
                content=None,
            ),
        ]


class EmptyTableSegmenter:
    tool_name = "empty_seg"

    def segment(self, pdf_path: Path) -> list[Region]:
        return [
            Region(
                page=0, bbox=[0, 0, 100, 50],
                region_type=ElementType.TABLE, confidence=0.9,
                content=ElementContent(),
            ),
        ]


def test_pipeline_drops_empty_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    pipe = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=EmptyTextSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=out,
        confidence_threshold=0.3,
    )
    pipe.run(pdf_path)
    data = json.loads((out / "content_list.json").read_text())
    types = sorted(e["type"] for e in data["elements"])
    # Under the current (pre-Task 8) merge rule the segmenter's empty content
    # flows through unchanged. _is_droppable must remove both empty-text
    # regions; the figure survives because the descriptor gives it a non-empty
    # description. Task 8 Step 5 revisits this test when the merge rule flips.
    assert types == ["figure"]


def test_pipeline_drops_table_without_markdown_or_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"

    class NoopTable:
        tool_name = "empty_seg"
        def extract(self, region_image, page_number):
            return ElementContent()

    pipe = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=EmptyTableSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=NoopTable(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=out,
        confidence_threshold=0.3,
    )
    pipe.run(pdf_path)
    data = json.loads((out / "content_list.json").read_text())
    assert not any(e["type"] == "table" for e in data["elements"])
```

- [ ] **Step 2: Run — expect FAIL**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py::test_pipeline_drops_empty_text extraction/tests/test_pipeline.py::test_pipeline_drops_table_without_markdown_or_text -v`
Expected: FAIL — currently empty content writes element anyway.

- [ ] **Step 3: Add `_is_droppable` and wire into `run`**

In `extraction/pipeline.py`, add:

```python
    def _is_droppable(self, region_type: ElementType, content: ElementContent) -> bool:
        if region_type in _TEXT_TYPES:
            return not (content.text or "").strip()
        if region_type == ElementType.TABLE:
            return not (content.markdown or content.text)
        if region_type == ElementType.FORMULA:
            return not (content.latex or content.text)
        # Visual types: kept as long as pipeline will supply image_path later.
        return False
```

And in the element loop, add the drop check:

```python
            content = self._extract_region(region, page_images)
            if content is None:
                continue
            if self._is_droppable(region.region_type, content):
                continue
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + checks**

Run: `venv/bin/pytest -q && venv/bin/ruff check extraction && venv/bin/mypy`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add extraction/pipeline.py extraction/tests/test_pipeline.py
git commit -m "feat(extraction): drop empty-content elements before persistence"
```

---

## Task 8: Merge rule rewrite — Config dictates

**Files:**
- Modify: `extraction/pipeline.py` (`_extract_region`, `_extractor_for`)
- Test: `extraction/tests/test_pipeline.py`

**Context:** Spec Section 4. Layout (bbox, type, caption) always from segmenter. Content from configured role tool. Tool-match optimization only when role tool name == segmenter tool name. `extractor` field is the role tool name.

- [ ] **Step 1: Write failing tests for the four merge scenarios**

Append to `extraction/tests/test_pipeline.py`:

```python
class _TableSegSameName:
    """Segmenter that names itself 'shared' and provides table content."""
    tool_name = "shared"

    def segment(self, pdf_path: Path) -> list[Region]:
        return [
            Region(
                page=0, bbox=[10, 10, 100, 100],
                region_type=ElementType.TABLE, confidence=0.95,
                content=ElementContent(markdown="SEG-MD", text="seg text", caption="Table 1"),
            ),
        ]


class _TableExtractorSameName:
    """Table extractor that names itself 'shared' and should NOT be called."""
    tool_name = "shared"
    def __init__(self):
        self.called = False
    def extract(self, region_image, page_number):
        self.called = True
        return ElementContent(markdown="EXT-MD", text="ext text")


class _TableExtractorOther:
    tool_name = "other"
    def __init__(self):
        self.called = False
    def extract(self, region_image, page_number):
        self.called = True
        return ElementContent(markdown="EXT-MD", text="ext text")


def test_merge_tool_match_reuses_segmenter_content(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    table_ext = _TableExtractorSameName()
    pipe = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=_TableSegSameName(),
        text_extractor=MockTextExtractor(),
        table_extractor=table_ext,
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=out,
        confidence_threshold=0.3,
    )
    pipe.run(pdf_path)
    data = json.loads((out / "content_list.json").read_text())
    table = next(e for e in data["elements"] if e["type"] == "table")
    assert table["content"]["markdown"] == "SEG-MD"
    assert table["content"]["caption"] == "Table 1"
    assert table["extractor"] == "shared"
    assert table_ext.called is False


def test_merge_tool_mismatch_runs_role_tool_and_discards_segmenter_content(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    table_ext = _TableExtractorOther()
    pipe = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=_TableSegSameName(),  # "shared"
        text_extractor=MockTextExtractor(),
        table_extractor=table_ext,  # "other"
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=out,
        confidence_threshold=0.3,
    )
    pipe.run(pdf_path)
    data = json.loads((out / "content_list.json").read_text())
    table = next(e for e in data["elements"] if e["type"] == "table")
    assert table["content"]["markdown"] == "EXT-MD"
    # Caption always from segmenter
    assert table["content"]["caption"] == "Table 1"
    assert table["extractor"] == "other"
    assert table_ext.called is True


class _FigureCaptionSegmenter:
    tool_name = "figseg"

    def segment(self, pdf_path: Path) -> list[Region]:
        return [
            Region(
                page=0, bbox=[0, 0, 200, 200],
                region_type=ElementType.FIGURE, confidence=0.9,
                content=ElementContent(caption="Fig. 1: The widget"),
            ),
        ]


def test_merge_figure_keeps_segmenter_caption_and_descriptor_description(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    pipe = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=_FigureCaptionSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),  # returns "A test figure"
        output_dir=out,
        confidence_threshold=0.3,
    )
    pipe.run(pdf_path)
    data = json.loads((out / "content_list.json").read_text())
    fig = next(e for e in data["elements"] if e["type"] == "figure")
    assert fig["content"]["caption"] == "Fig. 1: The widget"
    assert fig["content"]["description"] == "A test figure"
    assert fig["extractor"] == "mock_fig"
```

- [ ] **Step 2: Run — expect FAIL on most**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py -k merge -v`
Expected: FAIL on at least the mismatch-case and the figure-caption-preserved case.

- [ ] **Step 3: Rewrite `_extract_region` and `_extractor_for`**

Replace `_extract_region` in `extraction/pipeline.py`:

```python
    def _extract_region(
        self, region: Region, page_images: list[Image]
    ) -> ElementContent | None:
        if region.page < 0 or region.page >= len(page_images):
            return None

        role_tool_name = self._role_tool_name(region.region_type)
        segmenter_content = region.content

        tool_match = (
            role_tool_name == self.segmenter.tool_name
            and segmenter_content is not None
        )
        if tool_match:
            assert segmenter_content is not None
            content = segmenter_content.model_copy()
        else:
            content = self._run_role_tool(region, page_images)

        # Layout (caption) always from segmenter
        if segmenter_content and segmenter_content.caption:
            content.caption = segmenter_content.caption

        return content

    def _run_role_tool(
        self, region: Region, page_images: list[Image]
    ) -> ElementContent:
        page_img = page_images[region.page]
        if region.region_type in _TEXT_TYPES:
            return self.text_extractor.extract(page_img, region.page)
        writer = OutputWriter(self.output_dir)
        crop = writer.crop_region(page_img, region.bbox, dpi=self.dpi)
        if region.region_type == ElementType.TABLE:
            return self.table_extractor.extract(crop, region.page)
        if region.region_type == ElementType.FORMULA:
            return self.formula_extractor.extract(crop, region.page)
        # Visual types
        description = self.figure_descriptor.describe(crop)
        return ElementContent(description=description)
```

Replace `_extractor_for` with `_role_tool_name`:

```python
    def _role_tool_name(self, region_type: ElementType) -> str:
        if region_type in _TEXT_TYPES:
            return self.text_extractor.tool_name
        if region_type == ElementType.TABLE:
            return self.table_extractor.tool_name
        if region_type == ElementType.FORMULA:
            return self.formula_extractor.tool_name
        return self.figure_descriptor.tool_name
```

And update the call site in `run()`:

```python
            extractor_name = self._role_tool_name(region.region_type)
```

(Delete the old `_extractor_for` method entirely.)

- [ ] **Step 4: Run tests**

Run: `venv/bin/pytest extraction/tests/test_pipeline.py -v`
Expected: PASS across all merge-related tests.

- [ ] **Step 5: Revisit Task 7 empty-text test**

The `test_pipeline_drops_empty_text` test from Task 7 expected the figure to survive. Under the new merge rule, the two empty-text regions from `EmptyTextSegmenter` will match the mock text extractor only if their tool names line up. With `EmptyTextSegmenter.tool_name = "empty_seg"` and `MockTextExtractor.tool_name = "mock_ocr"`, the extractor runs and returns "Extracted text from page" — so the text regions survive. Adjust the test to pin this semantics: use a text extractor that also returns empty text:

```python
class _EmptyTextExtractor:
    tool_name = "empty_seg"
    def extract(self, page_image, page_number):
        return ElementContent(text="")

def test_pipeline_drops_empty_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    pipe = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=EmptyTextSegmenter(),
        text_extractor=_EmptyTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=out,
        confidence_threshold=0.3,
    )
    pipe.run(pdf_path)
    data = json.loads((out / "content_list.json").read_text())
    types = [e["type"] for e in data["elements"]]
    assert types == ["figure"]  # only the figure survives; both empty texts dropped
```

Remove the now-unused `_text_nonempty` helper.

- [ ] **Step 6: Run full suite + checks**

Run: `venv/bin/pytest -q && venv/bin/ruff check extraction && venv/bin/mypy`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add extraction/pipeline.py extraction/tests/test_pipeline.py
git commit -m "feat(extraction): merge rule — config dictates, pipeline obeys"
```

---

## Task 9: MinerU reads real confidence from layout_dets

**Files:**
- Modify: `extraction/adapters/mineru25_segmenter.py`
- Test: `extraction/tests/test_mineru25_segmenter.py` (new)

**Context:** Spec Section 6. MinerU `middle_json` carries `layout_dets[*].score`. Current adapter hardcodes `confidence=1.0`. Map scores to para_blocks by bbox match (exact equality on the integer-rounded bbox is enough — MinerU emits consistent bboxes for the same region across arrays).

- [ ] **Step 1: Skim MinerU middle_json shape**

Open `extraction/adapters/mineru25_segmenter.py` and review `_iter_para_blocks` and `_block_to_region`. Confirm each page dict in `raw["pdf_info"]` contains both `para_blocks` and `layout_dets`. If uncertain, look at a sample fixture:

Run: `find outputs -name "*_middle.json" | head -1 | xargs head -c 2000` (may be empty — fine, fall back to the docstring assumption).

- [ ] **Step 2: Write failing test with a minimal fixture**

Create `extraction/tests/test_mineru25_segmenter.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from extraction.adapters.mineru25_segmenter import (
    MinerU25Segmenter,
    _block_to_region,
    _confidence_for_block,
)
from extraction.models import ElementType


def test_confidence_for_block_matches_by_bbox() -> None:
    layout_dets = [
        {"bbox": [10, 20, 100, 50], "score": 0.42},
        {"bbox": [10, 60, 100, 200], "score": 0.91},
    ]
    block = {"bbox": [10, 60, 100, 200], "type": "text"}
    assert _confidence_for_block(block, layout_dets) == 0.91


def test_confidence_for_block_defaults_to_one_when_missing() -> None:
    layout_dets = [{"bbox": [0, 0, 1, 1], "score": 0.5}]
    block = {"bbox": [10, 20, 30, 40], "type": "text"}
    assert _confidence_for_block(block, layout_dets) == 1.0


def test_block_to_region_uses_layout_dets_confidence() -> None:
    layout_dets = [{"bbox": [0, 0, 100, 50], "score": 0.33}]
    block = {
        "bbox": [0, 0, 100, 50],
        "type": "text",
        "lines": [{"spans": [{"type": "text", "content": "hello"}]}],
    }
    region = _block_to_region(block, page_number=3, layout_dets=layout_dets)
    assert region is not None
    assert region.region_type == ElementType.TEXT
    assert abs(region.confidence - 0.33) < 1e-6
```

- [ ] **Step 3: Run — expect FAIL**

Run: `venv/bin/pytest extraction/tests/test_mineru25_segmenter.py -v`
Expected: ImportError on `_confidence_for_block` and signature mismatch on `_block_to_region`.

- [ ] **Step 4: Implement `_confidence_for_block` and thread through**

In `extraction/adapters/mineru25_segmenter.py`:

Add helper at module scope:

```python
def _confidence_for_block(
    block: dict[str, Any], layout_dets: list[dict[str, Any]]
) -> float:
    block_bbox = _to_bbox(block.get("bbox"))
    if block_bbox is None:
        return 1.0
    target = tuple(round(v) for v in block_bbox)
    for det in layout_dets:
        det_bbox = _to_bbox(det.get("bbox"))
        if det_bbox is None:
            continue
        if tuple(round(v) for v in det_bbox) == target:
            return float(det.get("score", 1.0))
    return 1.0
```

Change `_block_to_region` signature to accept `layout_dets` and replace every hardcoded `confidence=1.0` with `confidence=_confidence_for_block(block, layout_dets)`:

```python
def _block_to_region(
    block: dict[str, Any],
    page_number: int,
    layout_dets: list[dict[str, Any]],
) -> Region | None:
    block_type = block.get("type")
    bbox = _to_bbox(block.get("bbox"))
    if bbox is None:
        return None

    confidence = _confidence_for_block(block, layout_dets)

    if block_type == _BLOCK_TEXT:
        text = _lines_to_text(block.get("lines", []))
        if not text:
            return None
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=ElementType.TEXT,
            confidence=confidence,
            content=ElementContent(text=text),
        )
    # Repeat the same substitution in every other arm: HEADING, TABLE,
    # INTERLINE_EQUATION, IMAGE/CHART — each returns Region(..., confidence=confidence, ...).
```

Apply the substitution to every `Region(...)` construction in `_block_to_region`.

Change `_iter_para_blocks` to yield layout_dets alongside blocks:

```python
def _iter_para_blocks(
    raw: dict[str, Any],
) -> Iterator[tuple[int, dict[str, Any], list[dict[str, Any]]]]:
    for page_idx, page in enumerate(raw.get("pdf_info", [])):
        page_number = int(page.get("page_idx", page_idx))
        layout_dets = page.get("layout_dets") or []
        for block in page.get("para_blocks") or []:
            yield page_number, block, layout_dets
```

Update `MinerU25Segmenter.segment` to pass layout_dets:

```python
        regions: list[Region] = []
        for page_number, block, layout_dets in _iter_para_blocks(raw):
            region = _block_to_region(block, page_number, layout_dets)
            if region is not None:
                regions.append(region)
        return regions
```

- [ ] **Step 5: Run tests**

Run: `venv/bin/pytest extraction/tests/test_mineru25_segmenter.py -v`
Expected: PASS.

- [ ] **Step 6: Full suite + checks**

Run: `venv/bin/pytest -q && venv/bin/ruff check extraction && venv/bin/mypy`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add extraction/adapters/mineru25_segmenter.py extraction/tests/test_mineru25_segmenter.py
git commit -m "feat(extraction): MinerU adapter reads real confidence from layout_dets"
```

---

## Task 10: Cleanups — single reading_order, single OutputWriter

**Files:**
- Modify: `extraction/pipeline.py`

**Context:** Spec Section 10. `reading_order_index` is authoritatively set in `OutputWriter.build_content_list`, so the pipeline's pre-set is redundant. `OutputWriter(self.output_dir)` is instantiated once per call in the current merge code — after Task 8 we already use `writer` for most things, but the per-region `OutputWriter(...)` calls in `_run_role_tool` should reuse the writer already in `run()`.

- [ ] **Step 1: Remove redundant reading_order reassignment in `run()`**

In `extraction/pipeline.py`, delete the block:

```python
        # 6. Reassign reading order after filtering
        for idx, el in enumerate(elements):
            el.reading_order_index = idx
```

`build_content_list` already re-numbers globally across all sidecars. The per-element `reading_order_index` written to the sidecar before that call is a temporary seed; `build_content_list` overwrites it deterministically.

- [ ] **Step 2: Pass `writer` into `_run_role_tool`**

Change `_extract_region` and `_run_role_tool` to accept the shared `writer`:

```python
    def _extract_region(
        self,
        region: Region,
        page_images: list[Image],
        writer: OutputWriter,
    ) -> ElementContent | None:
        ...
        if tool_match:
            ...
        else:
            content = self._run_role_tool(region, page_images, writer)
        ...

    def _run_role_tool(
        self,
        region: Region,
        page_images: list[Image],
        writer: OutputWriter,
    ) -> ElementContent:
        page_img = page_images[region.page]
        if region.region_type in _TEXT_TYPES:
            return self.text_extractor.extract(page_img, region.page)
        crop = writer.crop_region(page_img, region.bbox, dpi=self.dpi)
        if region.region_type == ElementType.TABLE:
            return self.table_extractor.extract(crop, region.page)
        if region.region_type == ElementType.FORMULA:
            return self.formula_extractor.extract(crop, region.page)
        description = self.figure_descriptor.describe(crop)
        return ElementContent(description=description)
```

Update the `run()` call site:

```python
            content = self._extract_region(region, page_images, writer)
```

- [ ] **Step 3: Run tests + checks**

Run: `venv/bin/pytest -q && venv/bin/ruff check extraction && venv/bin/mypy`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add extraction/pipeline.py
git commit -m "refactor(extraction): share OutputWriter and drop redundant reading_order pass"
```

---

## Task 11: Update `docs/extraction_output.md`

**Files:**
- Modify: `docs/extraction_output.md`

**Context:** Spec's "Dokumentation" section lists four updates: new Koordinatensystem section, revised Merge-Regeln, revised IDs, tool-match + Phase 2 notes.

- [ ] **Step 1: Add Koordinatensystem section**

After the "Reading Order" section (line ~186) and before "IDs", insert:

```markdown
## Koordinatensystem

`bbox` in jedem `Region` und jedem `Element` ist in **PDF-Points**, Origin
oben-links, DPI-unabhängig. Beide mitgelieferten Segmenter (MinerU 2.5 via
`middle_json`, PyMuPDF via `get_text("dict")`) liefern bereits in Points.

Die Pipeline skaliert bbox nur beim Cropping auf das gerenderte Seitenbild:
`scale = dpi / 72`. Passiert an genau einer Stelle (`OutputWriter.crop_region`).
Negative Werte und Überläufe werden auf Bildgrenzen geklemmt, damit ein leicht
überschießendes bbox keinen kaputten Crop erzeugt.

Vorteil: Segmenter bleibt DPI-agnostisch. Wenn später mit anderer DPI
re-rendert wird, braucht weder Segmenter noch Output-Format geändert werden —
nur der eine Skalierungsschritt.
```

- [ ] **Step 2: Rewrite Merge-Regeln section**

Replace the existing "Merge-Regeln" block with:

```markdown
## Merge-Regeln

Die Pipeline entscheidet pro Region **nicht selbst**, welches Tool den Content
liefert. Die Config weist jedem Role (text, table, formula, figure) genau ein
Tool zu. Die Pipeline gehorcht.

Datenquellen pro Feld:

| Feld                                       | Quelle                  |
|--------------------------------------------|-------------------------|
| `bbox`, `page`, `type`, `reading_order`    | Segmenter               |
| `caption`                                  | Segmenter (Layout)      |
| `image_path`                               | Pipeline (aus dem Crop) |
| `text`                                     | `text_extractor`        |
| `markdown`                                 | `table_extractor`       |
| `latex`                                    | `formula_extractor`     |
| `description`                              | `figure_descriptor`     |

Ablauf pro Region:

1. Pipeline bestimmt das Role-Tool für den Region-Typ aus der Config.
2. Wenn `role_tool.tool_name == segmenter.tool_name`: Pipeline übernimmt
   `region.content` (Tool-Match-Optimierung — gleiches Tool, kein Re-Run nötig).
   Sonst: Pipeline cropped das Seitenbild und ruft das Role-Tool.
3. `caption` aus dem Segmenter bleibt in jedem Fall erhalten.
4. Ein Role-Tool-Output mit leerem Pflichtfeld (leerer Text, kein
   Markdown/Text bei Tabellen, kein LaTeX/Text bei Formeln) wird als
   „keine Daten" gewertet und das Element gedroppt. Visuelle Elemente
   dürfen persistieren, solange ein `image_path` vorhanden ist.

Beispiele:

| Config                                           | Verhalten                                                                                 |
|--------------------------------------------------|-------------------------------------------------------------------------------------------|
| `segmenter: mineru25`, `table_extractor: mineru25` | Tool-Match — MinerUs internes Markdown wird direkt übernommen.                          |
| `segmenter: mineru25`, `table_extractor: anderes` | Crop wird durch `anderes` geschickt, MinerUs Markdown verworfen. Caption bleibt erhalten. |
| `segmenter: mineru25`, `figure_descriptor: noop` | Figure-Element hat `caption` aus dem Segmenter, aber keine `description`.               |

Welche Tool-Kombination pro Role optimal ist, wird durch Benchmarks
entschieden (siehe `backlog.md` → „Per-role extractor benchmark"), nicht durch
Pipeline-Heuristik. Der Default `table_extractor: mineru25` nutzt die
Tool-Match-Optimierung — das ist Effizienz, kein Design-Constraint.
```

- [ ] **Step 3: Rewrite the IDs section**

Replace:

```markdown
- `element_id`: SHA-256 über `"<pdf_path>:<page>:<region_type>:<seq>"`, auf 16
  Hex-Zeichen gekürzt. Deterministisch — derselbe PDF produziert dieselben IDs.
```

with:

```markdown
- `element_id`: SHA-256 über
  `"<doc_id>:<page>:<region_type>:<round(x0)>,<round(y0)>,<round(x1)>,<round(y1)>"`,
  auf 16 Hex-Zeichen gekürzt. Pfadunabhängig: derselbe PDF-Inhalt liefert
  überall dieselben IDs. Bbox-basiert statt sequenzbasiert — stabil, auch
  wenn ein ML-Segmenter die Regionen nicht exakt in derselben Reihenfolge
  liefert.
```

- [ ] **Step 4: Add Phase-2 note on document_rich.json fail-safe**

At the end of the "Was Phase 2 bringt" section, append:

```markdown
Sobald `document_rich.json` in Phase 2 produziert wird, wird der
Output-Isolation-Fail-Safe in `Pipeline.run()` es als geprüftes Artefakt
aufnehmen — mit genau derselben Semantik wie `content_list.json` und
`segmentation.json`.
```

- [ ] **Step 5: Commit**

```bash
git add docs/extraction_output.md
git commit -m "docs(extraction): coordinate system, merge rule rewrite, bbox-based IDs"
```

---

## Task 12: Create `docs/writing_adapters.md`

**Files:**
- Create: `docs/writing_adapters.md`

**Context:** Spec Section 6 + "Dokumentation": checklist for adapter authors, so future adapters don't reintroduce hardcoded `1.0` confidence.

- [ ] **Step 1: Create the file**

Write `docs/writing_adapters.md`:

```markdown
# Writing Extraction Adapters

Checkliste für alle neuen Adapter (Renderer, Segmenter, Extractor,
Figure-Descriptor) im Extraction-Block.

## Pflicht

1. **Echte Confidence liefern (Segmenter).**
   Das ML-Modell liefert fast immer pro Region einen Score. Diesen in
   `Region.confidence` eintragen. **Nicht** hart auf `1.0` setzen — damit
   wird `confidence_threshold` in `ExtractionConfig` wirkungslos. Regelbasierte
   Segmenter (z. B. `PyMuPDFTextSegmenter`) dürfen `1.0` verwenden, müssen
   diesen Umstand aber im Adapter kommentieren.

2. **bbox in PDF-Points (Origin top-left).**
   Niemand im Projekt rechnet einen Pixel-bbox zurück in Points. Wenn das
   Tool Pixel liefert, muss der Adapter selbst umrechnen. Das Cropping auf
   Pixel passiert an genau einer Stelle (`OutputWriter.crop_region`) mit
   `scale = dpi / 72`.

3. **`tool_name` konsistent.**
   Property `tool_name` und Registry-Name (`@register_xxx("NAME")`) müssen
   identisch sein. Die Pipeline vergleicht diese Strings für die
   Tool-Match-Optimierung beim Merge.

4. **Merge-Regel beachten (Extractor-Rollen).**
   Ein Extractor gibt nur Content-Felder zurück (`text`, `markdown`,
   `latex`, `description`). Layout-Felder (`caption`) gehören dem Segmenter
   und werden vom Extractor-Output überschrieben, falls er sie setzt.

## Empfohlen

- Lazy Imports für schwere Dependencies (Torch, Transformers, MinerU):
  Import im `__init__` oder in einem `_load()`-Helper, nicht auf Modul-Ebene.
  Die Pipeline registriert Adapter beim Import — schwere Importe dort
  machen `python -m extraction` auf CPU-Hosts unnötig langsam/fragil.
- Tests in `extraction/tests/test_<adapter>.py`. Integration-Tests, die
  GPU oder Modell-Downloads brauchen, mit `@pytest.mark.integration`
  markieren.

## Beispiele

- Typischer Content-Extractor (CPU, keine ML-Confidence):
  `extraction/adapters/stubs.py`
- Rich Segmenter (ML, echte Scores):
  `extraction/adapters/mineru25_segmenter.py`
- Regelbasierter Segmenter (keine ML-Confidence):
  `extraction/adapters/pymupdf_text_segmenter.py`
```

- [ ] **Step 2: Commit**

```bash
git add docs/writing_adapters.md
git commit -m "docs(extraction): adapter-author checklist"
```

---

## Task 13: Update `README.md`

**Files:**
- Modify or Create: `README.md`

**Context:** Spec "Dokumentation": install commands, GPU extra, CPU config example, commands. Check if README.md already exists and update it instead of rewriting.

- [ ] **Step 1: Check current README**

Run: `test -f README.md && wc -l README.md || echo "missing"`
Record the current state before editing. If it already has project-specific content, preserve it and only extend the install / usage sections.

- [ ] **Step 2: Write or extend `README.md`**

If no README exists, write:

```markdown
# techpdfparser

Technical-PDF parsing into a unified structured output: text, tables,
formulas, figures, diagrams, drawings. Output format is a stable contract
(see `docs/extraction_output.md`). Tools are swappable per role via YAML.

## Install

CPU-only (no segmenter):

    pip install -e .

Full GPU stack (MinerU 2.5, OlmOCR, Qwen2.5-VL):

    pip install -e .[gpu]

PyTorch: CUDA builds depend on your system. Follow
https://pytorch.org/get-started/locally/ and install a matching
`torch` wheel before or alongside `-e .[gpu]`.

## CLI

Extract a PDF:

    python -m extraction extract path/to/document.pdf --config config.yaml --output outputs/run1/

Rebuild `content_list.json` from existing sidecars (no re-extraction):

    python -m extraction rebuild outputs/run1/

## CPU-only config example

    extraction:
      renderer: pymupdf
      segmenter: pymupdf_text
      text_extractor: noop
      table_extractor: noop
      formula_extractor: noop
      figure_descriptor: noop
      output_dir: outputs
      dpi: 150

## Quality gates

    pytest -q
    ruff check extraction
    mypy

Integration tests (MinerU / GPU) are marker-gated:

    pytest -m integration
```

If a README already exists, insert or replace the "Install", "CLI", "CPU-only
config example", and "Quality gates" sections with the content above. Keep
any project-specific narrative that's already there.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): install, CLI, CPU-config example"
```

---

## Task 14: Backlog entries (English)

**Files:**
- Modify: `backlog.md`

**Context:** Spec "Backlog-Einträge" — two new entries in English (convention per user feedback). Extend the existing benchmark entry, add the bbox-overlap-dedup entry.

- [ ] **Step 1: Append both entries to `backlog.md`**

Append:

```markdown

---

## 2026-04-18 — Bbox-overlap Deduplication

**Priority:** Nice-to-have (defensive)

### User Story
As a pipeline maintainer, I want overlapping regions from a single segmenter
to be collapsed so that the same content is not written twice (once as a
table, once as a figure, for example).

### Tasks
- [ ] Add a post-segmentation step that computes IoU between pairs of
      regions on the same page.
- [ ] For pairs with IoU > 0.8, drop the lower-confidence region.
- [ ] Log dropped regions for inspection.

### Acceptance Criteria
- [ ] Overlapping regions on the same page are deduplicated before
      extraction runs.
- [ ] Unit tests cover non-overlap, partial overlap, full overlap.
- [ ] Feature is off by default until evidence of duplicates in real runs.

### Notes
Parked until duplicates are observed in real PDF runs. Threshold 0.8 is a
starting point; revisit once benchmark corpus exists.
```

And extend the existing 2026-04-16 benchmark entry with a new task bullet:

Edit the existing "2026-04-16 — Compare PDF Extraction Tools" block's
`### Tasks` list — append:

```markdown
- [ ] Run the extraction pipeline with different `table_extractor`,
      `formula_extractor`, `figure_descriptor` combinations on the
      corpus. Produce a per-role recommendation table. Unblocks
      data-driven default changes in `ExtractionConfig`.
```

- [ ] **Step 2: Commit**

```bash
git add backlog.md
git commit -m "chore(backlog): bbox-overlap dedup and per-role extractor benchmark"
```

---

## Task 15: Archive stale `tasks/` files

**Files:**
- Modify: `tasks/extraction_completion_plan.md`
- Modify: `tasks/todo.md`

**Context:** Spec "Dokumentation": both files reference the old `src/...` layout. Don't delete — mark as archive so the history stays readable.

- [ ] **Step 1: Prepend an ARCHIVED banner to each file**

At the top of `tasks/extraction_completion_plan.md`:

```markdown
> **ARCHIVED — 2026-04-18.** References the pre-refactor `src/…` layout.
> Current extraction lives under `extraction/`. Kept for history only.
> See `docs/extraction_output.md` and
> `docs/superpowers/specs/2026-04-18-extraction-hardening-design.md`.

```

Same banner at the top of `tasks/todo.md`.

- [ ] **Step 2: Commit**

```bash
git add tasks/extraction_completion_plan.md tasks/todo.md
git commit -m "docs(tasks): mark pre-refactor plans as archived"
```

---

## Task 16: Quality gates + CPU smoke test

**Files:** none — verification only.

**Context:** Spec "Test Plan": full pytest / ruff / mypy green; CPU smoke on a real PDF.

- [ ] **Step 1: Run all quality gates**

Run: `venv/bin/pytest -q && venv/bin/ruff check extraction && venv/bin/mypy`
Expected: all green.

- [ ] **Step 2: CPU smoke test on a real PDF**

Pick one of the test PDFs at the repo root (`1.9.20 PV 1001.12, Rev. 3.pdf` is small):

Create a throwaway CPU config, e.g. `/tmp/cpu_smoke.yaml`:

```yaml
extraction:
  renderer: pymupdf
  segmenter: pymupdf_text
  text_extractor: noop
  table_extractor: noop
  formula_extractor: noop
  figure_descriptor: noop
  output_dir: /tmp/smoke_out
  dpi: 150
```

Run:

```bash
rm -rf /tmp/smoke_out
venv/bin/python -m extraction extract "1.9.20 PV 1001.12, Rev. 3.pdf" \
    --config /tmp/cpu_smoke.yaml --output /tmp/smoke_out
```

Expected:
- Prints `Elements: N` and `Pages: M` (both > 0)
- `/tmp/smoke_out/content_list.json` exists
- `/tmp/smoke_out/segmentation.json` exists
- `/tmp/smoke_out/pages/0/page.png` exists

Inspect content_list briefly:

```bash
venv/bin/python -c "
import json
d = json.load(open('/tmp/smoke_out/content_list.json'))
print('source_file:', d['source_file'])
print('doc_id:', d['doc_id'])
print('elements:', len(d['elements']))
print('first element:', d['elements'][0] if d['elements'] else '<none>')
"
```

Confirm `source_file` is the filename (no slashes), element_ids are 16-hex.

- [ ] **Step 3: Re-run aborts (fail-safe check)**

Run the same extract command again without removing `/tmp/smoke_out`:

```bash
venv/bin/python -m extraction extract "1.9.20 PV 1001.12, Rev. 3.pdf" \
    --config /tmp/cpu_smoke.yaml --output /tmp/smoke_out
```

Expected: traceback with `FileExistsError` mentioning `content_list.json`.

- [ ] **Step 4: No changes to commit** — this is a verification step.

---

## Summary of Commits (expected)

1. `chore(packaging): pin pymupdf runtime dep and explicit package discovery`
2. `fix(extraction): store filename only in content_list.source_file`
3. `feat(extraction): wire top-level dpi to renderer and pipeline`
4. `fix(extraction): scale PDF-point bboxes to pixels and clamp at crop time`
5. `fix(extraction): bbox-based element_id, path-independent`
6. `feat(extraction): fail-safe on non-empty output dir in Pipeline.run()`
7. `feat(extraction): drop empty-content elements before persistence`
8. `feat(extraction): merge rule — config dictates, pipeline obeys`
9. `feat(extraction): MinerU adapter reads real confidence from layout_dets`
10. `refactor(extraction): share OutputWriter and drop redundant reading_order pass`
11. `docs(extraction): coordinate system, merge rule rewrite, bbox-based IDs`
12. `docs(extraction): adapter-author checklist`
13. `docs(readme): install, CLI, CPU-config example`
14. `chore(backlog): bbox-overlap dedup and per-role extractor benchmark`
15. `docs(tasks): mark pre-refactor plans as archived`

Task 16 is verification-only; no commit.

After Task 16: push to origin.

```bash
git push origin master
```
