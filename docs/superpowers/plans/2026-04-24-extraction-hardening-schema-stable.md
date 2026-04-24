# Extraction-Härtung ohne Schema-Bruch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fünf latente Code-vs-Contract-Lücken in der Extraction-Pipeline schließen (Stage-2-Scope, TextExtractor-Semantik, FigureDescriptor-Caption-Kontext, `--out`-Default, Doku-Drift) — ohne Schema-Bruch am Output-Format.

**Architecture:** TDD, kleinschrittig. Jede Task: roter Test → minimale Implementierung → grüner Test → commit. Arbeit läuft auf eigener Branch `extraction-hardening-schema-stable`. Am Ende Quality-Gates (pytest, ruff, mypy), Merge nach `master`, Push.

**Tech Stack:** Python 3.x, pytest, ruff, mypy, Pydantic v2, PIL.

**Spec:** `docs/superpowers/specs/2026-04-24-extraction-hardening-schema-stable-design.md`

---

## Task 1: Branch aufsetzen und Design-Doc committen

**Files:**
- Create branch: `extraction-hardening-schema-stable`
- Commit: `docs/superpowers/specs/2026-04-24-extraction-hardening-schema-stable-design.md` + dieser Plan

- [ ] **Step 1: Git-Stand prüfen**

Run:
```bash
git status
git log -1 --oneline
```

Expected: `backlog.md` und `.directory` und die neuen docs-Dateien als untracked/modified. Letzter Commit `d53c37d feat(extraction): preserve raw table HTML alongside flattened markdown`.

- [ ] **Step 2: Branch anlegen und wechseln**

Run:
```bash
git checkout -b extraction-hardening-schema-stable
```

Expected: `Switched to a new branch 'extraction-hardening-schema-stable'`. Uncommitted Änderungen bleiben im Working Tree.

- [ ] **Step 3: Design-Doc und Plan committen**

Run:
```bash
git add docs/superpowers/specs/2026-04-24-extraction-hardening-schema-stable-design.md \
        docs/superpowers/plans/2026-04-24-extraction-hardening-schema-stable.md
git commit -m "docs(extraction): spec + plan für schema-stabile Härtung"
```

Expected: ein neuer Commit auf der Branch. `backlog.md` und `.directory` bleiben unangetastet (nicht mitstagen).

---

## Task 2: `TextExtractor`-Docstring auf Region-Crop umstellen

**Files:**
- Modify: `extraction/interfaces.py:40-42`

Das ist eine reine Docstring-Änderung; das tatsächliche Verhalten ändern wir in Task 4.

- [ ] **Step 1: Docstring anpassen**

In `extraction/interfaces.py`, ersetze den `TextExtractor`-Block (Zeile 36-42):

```python
class TextExtractor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        """Extract text from a region crop. Returns content with text field set.

        The image is a cropped region (heading/paragraph) produced by the
        pipeline from the rendered page image, not the full page.
        """
        ...
```

- [ ] **Step 2: Type-Check**

Run:
```bash
venv/bin/mypy extraction
```

Expected: keine neuen Fehler (Docstring-Änderung ist typ-neutral).

- [ ] **Step 3: Commit**

Run:
```bash
git add extraction/interfaces.py
git commit -m "docs(extraction): TextExtractor nimmt Region-Crop, nicht Vollseite"
```

---

## Task 3: Stage 2 — Sidecar-Existenz-Skip

**Files:**
- Modify: `extraction/stages/extract_text.py`
- Test: `extraction/tests/test_stages_extract_text.py`

Wenn Stage 1 im Role-Match-Pfad bereits ein Sidecar geschrieben hat, soll Stage 2 diese Region überspringen — unabhängig vom `.stages/extract-text.done`-Marker.

- [ ] **Step 1: Roter Test**

Am Ende von `extraction/tests/test_stages_extract_text.py` anhängen:

```python
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
    import json as _json
    data = _json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["extractor"] == "stub_segmenter"
    assert data["content"]["text"] == "from stage 1 passthrough"
```

- [ ] **Step 2: Test laufen, muss FAIL**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_extract_text.py::test_text_skips_region_when_sidecar_already_exists -v
```

Expected: FAIL — aktuelle Implementierung überschreibt das Sidecar mit dem Stub-Output.

- [ ] **Step 3: Skip-Logik implementieren**

In `extraction/stages/extract_text.py`, ersetze die Region-Schleife in `_process_one` (ab Zeile 95):

```python
def _process_one(
    out_dir: Path,
    writer: OutputWriter,
    meta: dict,
    extractor: object,
    cfg: ExtractionConfig,
) -> None:
    regions: list[Region] = meta["regions"]
    doc_id: str = meta["doc_id"]
    for region in regions:
        if region.region_type not in _TARGET_TYPES:
            continue
        if region.confidence < cfg.confidence_threshold:
            continue
        el_id = _element_id(doc_id, region)
        sidecar = (
            out_dir / "pages" / str(region.page)
            / f"{el_id}_{region.region_type.value}.json"
        )
        if sidecar.exists():
            continue
        page_img = _load_page(out_dir, region.page)
        content: ElementContent = extractor.extract(page_img, region.page)  # type: ignore[attr-defined]
        if region.content is not None and region.content.caption:
            content.caption = region.content.caption
        if not (content.text or "").strip():
            continue
        el = Element(
            element_id=el_id,
            type=region.region_type,
            page=region.page,
            bbox=region.bbox,
            reading_order_index=region.reading_order_index,
            section_path=[],
            confidence=region.confidence,
            extractor=extractor.tool_name,  # type: ignore[attr-defined]
            content=content,
        )
        writer.write_element_sidecar(el)
```

- [ ] **Step 4: Test laufen, muss PASS**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_extract_text.py -v
```

Expected: alle Tests PASS (bestehende Happy-Path-Tests haben keine Pre-Sidecars und laufen wie gehabt).

- [ ] **Step 5: Commit**

Run:
```bash
git add extraction/stages/extract_text.py extraction/tests/test_stages_extract_text.py
git commit -m "fix(extraction): Stage 2 überspringt Regions mit vorhandenem Sidecar"
```

---

## Task 4: Stage 2 — Region-Crop statt Vollseite

**Files:**
- Modify: `extraction/stages/extract_text.py`
- Test: `extraction/tests/test_stages_extract_text.py`

- [ ] **Step 1: Stub für Größen-Aufzeichnung + roten Test anhängen**

In `extraction/tests/test_stages_extract_text.py`:

```python
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
```

- [ ] **Step 2: Test laufen, muss FAIL**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_extract_text.py::test_text_extractor_receives_region_crop -v
```

Expected: FAIL — Assertions melden 600×800 (Vollseite).

- [ ] **Step 3: Crop einbauen**

In `extraction/stages/extract_text.py`, `_process_one` die Extractor-Call-Zeile ersetzen. Vor der Schleife das `writer.crop_region(...)` nutzen. Konkret: zwischen `page_img = _load_page(...)` und `content = extractor.extract(...)`:

```python
        page_img = _load_page(out_dir, region.page)
        crop = writer.crop_region(
            page_img, region.bbox, dpi=cfg.resolve_renderer_dpi()
        )
        content: ElementContent = extractor.extract(crop, region.page)  # type: ignore[attr-defined]
```

- [ ] **Step 4: Test laufen, muss PASS**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_extract_text.py -v
```

Expected: alle PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add extraction/stages/extract_text.py extraction/tests/test_stages_extract_text.py
git commit -m "fix(extraction): Stage 2 reicht Region-Crop an TextExtractor statt Vollseite"
```

---

## Task 5: Stage 2 — Table-Support

**Files:**
- Modify: `extraction/stages/extract_text.py`
- Test: `extraction/tests/test_stages_extract_text.py`

- [ ] **Step 1: Test-Stub + roter Test anhängen**

In `extraction/tests/test_stages_extract_text.py`:

```python
from extraction.registry import register_table_extractor


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
```

- [ ] **Step 2: Test laufen, muss FAIL**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_extract_text.py::test_table_role_mismatch_extracts_sidecar -v
```

Expected: FAIL — Stage 2 verarbeitet TABLE derzeit nicht, kein Sidecar.

- [ ] **Step 3: TABLE in `_TARGET_TYPES` aufnehmen und Role-Extractor-Lookup implementieren**

In `extraction/stages/extract_text.py`, Oberkante:

```python
from ..registry import get_formula_extractor, get_table_extractor, get_text_extractor
```

Ersetze `_TARGET_TYPES`:

```python
_TARGET_TYPES = {
    ElementType.TEXT, ElementType.HEADING,
    ElementType.TABLE, ElementType.FORMULA,
}
```

Ersetze den Extractor-Lookup-Block (aktuell `extractor = get_text_extractor(...)` in `run_text`):

```python
    text_extractor = get_text_extractor(
        cfg.text_extractor, **cfg.get_adapter_config(cfg.text_extractor)
    )
    table_extractor = get_table_extractor(
        cfg.table_extractor, **cfg.get_adapter_config(cfg.table_extractor)
    )
    formula_extractor = get_formula_extractor(
        cfg.formula_extractor, **cfg.get_adapter_config(cfg.formula_extractor)
    )
    extractors: dict[ElementType, object] = {
        ElementType.TEXT: text_extractor,
        ElementType.HEADING: text_extractor,
        ElementType.TABLE: table_extractor,
        ElementType.FORMULA: formula_extractor,
    }
```

Ersetze den Aufruf von `_process_one`:

```python
            _process_one(out_dir, writer, meta, extractors, cfg)
```

Und ersetze die Signatur + den Extractor-Call in `_process_one`:

```python
def _process_one(
    out_dir: Path,
    writer: OutputWriter,
    meta: dict,
    extractors: dict[ElementType, object],
    cfg: ExtractionConfig,
) -> None:
    regions: list[Region] = meta["regions"]
    doc_id: str = meta["doc_id"]
    for region in regions:
        if region.region_type not in _TARGET_TYPES:
            continue
        if region.confidence < cfg.confidence_threshold:
            continue
        el_id = _element_id(doc_id, region)
        sidecar = (
            out_dir / "pages" / str(region.page)
            / f"{el_id}_{region.region_type.value}.json"
        )
        if sidecar.exists():
            continue
        extractor = extractors[region.region_type]
        page_img = _load_page(out_dir, region.page)
        crop = writer.crop_region(
            page_img, region.bbox, dpi=cfg.resolve_renderer_dpi()
        )
        content: ElementContent = extractor.extract(crop, region.page)  # type: ignore[attr-defined]
        if region.content is not None and region.content.caption:
            content.caption = region.content.caption
        if region.region_type in (ElementType.TEXT, ElementType.HEADING):
            if not (content.text or "").strip():
                continue
        else:
            # TABLE / FORMULA: Crop + image_path werden in Task 7 ergänzt.
            if not (
                (content.text or "").strip()
                or (content.markdown or "").strip()
                or (content.latex or "").strip()
            ):
                continue
        el = Element(
            element_id=el_id,
            type=region.region_type,
            page=region.page,
            bbox=region.bbox,
            reading_order_index=region.reading_order_index,
            section_path=[],
            confidence=region.confidence,
            extractor=extractor.tool_name,  # type: ignore[attr-defined]
            content=content,
        )
        writer.write_element_sidecar(el)
```

- [ ] **Step 4: Test laufen, muss PASS**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_extract_text.py -v
```

Expected: alle PASS (inkl. neuer Table-Test).

- [ ] **Step 5: Commit**

Run:
```bash
git add extraction/stages/extract_text.py extraction/tests/test_stages_extract_text.py
git commit -m "feat(extraction): Stage 2 verarbeitet Table-Regions bei role-mismatch"
```

---

## Task 6: Stage 2 — Formula-Support

**Files:**
- Modify: (nichts mehr an Code — Logik aus Task 5 deckt Formula ab)
- Test: `extraction/tests/test_stages_extract_text.py`

Formulas sind in `_TARGET_TYPES` und im `extractors`-Dict seit Task 5 drin. Wir brauchen nur noch den passenden Test, um sicherzustellen, dass der Pfad greift.

- [ ] **Step 1: Test anhängen**

```python
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
```

- [ ] **Step 2: Test laufen, muss PASS**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_extract_text.py::test_formula_role_mismatch_extracts_sidecar -v
```

Expected: PASS (Logik aus Task 5 deckt Formula ab).

- [ ] **Step 3: Commit**

Run:
```bash
git add extraction/tests/test_stages_extract_text.py
git commit -m "test(extraction): Formula-Regions im Stage-2-role-mismatch-Pfad"
```

---

## Task 7: Stage 2 — Table/Formula persistieren auch mit leerem Content (image_path-Regel)

**Files:**
- Modify: `extraction/stages/extract_text.py`
- Test: `extraction/tests/test_stages_extract_text.py`

Per Spec: `table`/`formula` Sidecar persistiert, solange `image_path` gesetzt ist. Aktuell droppen wir in Task 5 bei leerem Content.

- [ ] **Step 1: Test anhängen — leerer Table-Content + Crop = Sidecar bleibt**

```python
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
```

- [ ] **Step 2: Test laufen, muss FAIL**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_extract_text.py::test_table_persists_with_image_path_when_content_empty -v
```

Expected: FAIL — Sidecar wird gedroppt, weil markdown/text/latex leer.

- [ ] **Step 3: Drop-Regel relaxen + Crop für table/formula schreiben**

In `extraction/stages/extract_text.py`, `_process_one` überarbeiten — ersetze den Block ab `crop = writer.crop_region(...)`:

```python
        crop = writer.crop_region(
            page_img, region.bbox, dpi=cfg.resolve_renderer_dpi()
        )
        content: ElementContent = extractor.extract(crop, region.page)  # type: ignore[attr-defined]
        if region.content is not None and region.content.caption:
            content.caption = region.content.caption
        if region.region_type in (ElementType.TEXT, ElementType.HEADING):
            if not (content.text or "").strip():
                continue
        else:
            # TABLE / FORMULA: Crop + image_path persistieren
            # auch wenn markdown/latex/text leer sind.
            rel = writer.save_element_crop(
                page=region.page, element_id=el_id,
                element_type=region.region_type.value, image=crop,
            )
            content.image_path = str(rel.relative_to(writer.output_dir))
        el = Element(
            element_id=el_id,
            type=region.region_type,
            page=region.page,
            bbox=region.bbox,
            reading_order_index=region.reading_order_index,
            section_path=[],
            confidence=region.confidence,
            extractor=extractor.tool_name,  # type: ignore[attr-defined]
            content=content,
        )
        writer.write_element_sidecar(el)
```

(Der vorherige Else-Zweig mit der leeren-Content-Drop-Logik wird komplett entfernt; stattdessen wird für TABLE/FORMULA immer der Crop geschrieben und das Sidecar persistiert.)

- [ ] **Step 4: Test laufen, muss PASS**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_extract_text.py -v
```

Expected: alle PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add extraction/stages/extract_text.py extraction/tests/test_stages_extract_text.py
git commit -m "fix(extraction): Table/Formula persistieren mit image_path auch bei leerem Content"
```

---

## Task 8: `FigureDescriptor.describe` mit optionalem `caption`-Parameter

**Files:**
- Modify: `extraction/interfaces.py:63-69`
- Modify: `extraction/adapters/stubs.py:58-67`
- Modify: `extraction/adapters/qwen25vl_figure.py:93-117`
- Modify: `extraction/tests/test_stages_describe_figures.py:22-23,31-33,41-43`

Der Interface-Vertrag kriegt einen optionalen `caption`-Parameter. Alle bestehenden Adapter und Test-Stubs werden compatible gemacht.

- [ ] **Step 1: Protocol-Signatur erweitern**

In `extraction/interfaces.py`, ersetze den `FigureDescriptor`-Block:

```python
class FigureDescriptor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def describe(self, image: Image, caption: str | None = None) -> str:
        """Generate a text description of a figure/diagram image.

        ``caption`` carries the Segmenter-detected figure caption when
        available. Adapters that can use it (VLMs) should treat it as
        grounding context; simple adapters may ignore it.
        """
        ...
```

- [ ] **Step 2: Noop-Stub anpassen**

In `extraction/adapters/stubs.py`, ersetze `NoopFigureDescriptor.describe`:

```python
    def describe(self, image: Image, caption: str | None = None) -> str:
        return ""
```

- [ ] **Step 3: Qwen-Descriptor-Signatur anpassen (Prompt-Nutzung folgt in Task 10)**

In `extraction/adapters/qwen25vl_figure.py`, Methoden-Signatur ersetzen:

```python
    def describe(self, image: Image, caption: str | None = None) -> str:
        self._load()
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _DESCRIBE_PROMPT},
                ],
            }
        ]
```

(Wir nutzen `caption` noch nicht — dieser Task zieht nur die Signatur durch. Prompt-Änderung ist Task 10.)

- [ ] **Step 4: Test-Stubs in `test_stages_describe_figures.py` anpassen**

Ersetze in `extraction/tests/test_stages_describe_figures.py` die drei Stub-`describe`-Methoden:

```python
    def describe(self, image, caption=None):
        return f"a stub description {image.size}"
```

```python
    def describe(self, image, caption=None):
        return ""
```

```python
    def describe(self, image, caption=None):
        raise RuntimeError("describe blew up")
```

- [ ] **Step 5: Alle Tests laufen**

Run:
```bash
venv/bin/pytest extraction/tests/ -v
```

Expected: alle PASS (Signatur-Erweiterung ist rückwärtskompatibel).

- [ ] **Step 6: Commit**

Run:
```bash
git add extraction/interfaces.py extraction/adapters/stubs.py \
        extraction/adapters/qwen25vl_figure.py \
        extraction/tests/test_stages_describe_figures.py
git commit -m "feat(extraction): FigureDescriptor.describe akzeptiert optionalen caption-Kontext"
```

---

## Task 9: Stage 3 reicht Caption an den Descriptor durch

**Files:**
- Modify: `extraction/stages/describe_figures.py:104`
- Test: `extraction/tests/test_stages_describe_figures.py`

- [ ] **Step 1: Roten Test anhängen**

In `extraction/tests/test_stages_describe_figures.py`:

```python
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


def test_figures_passes_caption_to_describer(tmp_path: Path):
    """Stage 3 reicht region.content.caption durch an describe()."""
    _captured_captions.clear()
    out = tmp_path / "doc1"
    _seed_segment(out)  # seed hat FIGURE mit caption="Fig 1", DIAGRAM ohne
    cfg = _cfg(figure_descriptor="stub_fig_capture")
    assert run_figures([out], cfg) == 0
    assert "Fig 1" in _captured_captions
    assert None in _captured_captions
```

- [ ] **Step 2: Test laufen, muss FAIL**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_describe_figures.py::test_figures_passes_caption_to_describer -v
```

Expected: FAIL — `_captured_captions` enthält nur `None` (aktuell wird caption nicht durchgereicht).

- [ ] **Step 3: Caption durchreichen**

In `extraction/stages/describe_figures.py`, in `_process_one` die `describe`-Aufrufzeile (aktuell Zeile 104) ersetzen:

```python
        region_caption = (
            region.content.caption
            if region.content is not None
            else None
        )
        description = describer.describe(crop, caption=region_caption)  # type: ignore[attr-defined]
```

- [ ] **Step 4: Test laufen, muss PASS**

Run:
```bash
venv/bin/pytest extraction/tests/test_stages_describe_figures.py -v
```

Expected: alle PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add extraction/stages/describe_figures.py extraction/tests/test_stages_describe_figures.py
git commit -m "feat(extraction): Stage 3 reicht Region-Caption an FigureDescriptor durch"
```

---

## Task 10: Qwen-Prompt nutzt Caption als Grounding-Kontext

**Files:**
- Modify: `extraction/adapters/qwen25vl_figure.py:20-25,93-105`
- Test: kein neuer Test (Qwen läuft nur auf GPU; Prompt-Konstruktion wird als Unit getestet)

- [ ] **Step 1: Prompt-Konstruktion testbar machen + roter Test**

Füge eine neue Datei `extraction/tests/test_qwen25vl_prompt.py` hinzu:

```python
"""Unit-Tests für die Prompt-Konstruktion des Qwen-Descriptors.

Lädt das Model nicht — testet nur die Text-Assembly.
"""
from __future__ import annotations

from extraction.adapters.qwen25vl_figure import _build_prompt


def test_prompt_without_caption_is_image_only():
    prompt = _build_prompt(caption=None)
    assert "caption" not in prompt.lower()
    assert "describe this figure" in prompt.lower()


def test_prompt_with_caption_includes_it_as_grounding():
    prompt = _build_prompt(caption="Figure 2. Tensile test specimens.")
    assert "Figure 2. Tensile test specimens." in prompt
    # Das Prompt soll explizit sagen: nichts erfinden, Caption als Kontext.
    assert "do not invent" in prompt.lower() or "nicht erfinden" in prompt.lower()


def test_prompt_with_empty_caption_treated_as_none():
    assert _build_prompt(caption="") == _build_prompt(caption=None)
    assert _build_prompt(caption="   ") == _build_prompt(caption=None)
```

- [ ] **Step 2: Test laufen, muss FAIL**

Run:
```bash
venv/bin/pytest extraction/tests/test_qwen25vl_prompt.py -v
```

Expected: FAIL — `_build_prompt` existiert noch nicht.

- [ ] **Step 3: `_build_prompt` einführen und im `describe` nutzen**

In `extraction/adapters/qwen25vl_figure.py`, ersetze den Prompt-Block (Zeile 20-25) und die `describe`-Methode (Zeile 93-117):

```python
_DESCRIBE_PROMPT_BASE = (
    "Describe this figure from a technical document concisely. "
    "Identify the visualization type (chart, diagram, graph, schematic, etc.), "
    "what data or concept it shows, and any key values or trends visible. "
    "Be specific and technical. Two to four sentences maximum."
)


def _build_prompt(caption: str | None) -> str:
    """Return the Qwen user-text, with caption grounding when non-empty."""
    if caption is None or not caption.strip():
        return _DESCRIBE_PROMPT_BASE
    # Concatenation (not .format) — der Caption-Text darf `{}` enthalten.
    return (
        _DESCRIBE_PROMPT_BASE
        + "\n\nThe figure has this caption from the source document:\n"
        + "  " + caption.strip() + "\n\n"
        + "Use the caption as grounding context. Do not invent any fact "
        + "that is not supported by either the image or the caption."
    )
```

In `describe` den Prompt-Aufruf anpassen:

```python
    def describe(self, image: Image, caption: str | None = None) -> str:
        self._load()
        import torch

        prompt_text = _build_prompt(caption)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self._runtime_device)

        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=256)

        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()
```

Entferne die alte `_DESCRIBE_PROMPT`-Konstante (durch `_DESCRIBE_PROMPT_BASE` ersetzt).

- [ ] **Step 4: Test laufen, muss PASS**

Run:
```bash
venv/bin/pytest extraction/tests/test_qwen25vl_prompt.py -v
```

Expected: alle PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add extraction/adapters/qwen25vl_figure.py extraction/tests/test_qwen25vl_prompt.py
git commit -m "feat(extraction): Qwen-Prompt nutzt Caption als Grounding-Kontext"
```

---

## Task 11: olmocr2-Prompt auf Region-Crop umstellen

**Files:**
- Modify: `extraction/adapters/olmocr2_text.py:1-35,104-107`

Das Modell läuft nur mit GPU; keine End-to-End-Tests. Wir aktualisieren Docstring und Prompt, damit sie den semantischen Vertrag (Crop) widerspiegeln.

- [ ] **Step 1: Modul-Docstring und Prompt anpassen**

In `extraction/adapters/olmocr2_text.py`, Modul-Docstring ersetzen:

```python
"""olmOCR-2 text extractor.

Runs the allenai/olmOCR-2 Vision2Seq model over a cropped region image
and returns markdown text. The pipeline crops the region from the
rendered page before calling this adapter — the crop carries exactly
the text block, heading, or caption that the segmenter detected.

Model: allenai/olmOCR-2-7B-1025
Requires: pip install transformers torch olmocr
"""
```

Ersetze `_OLMOCR_PROMPT`:

```python
_OLMOCR_PROMPT = (
    "Attached is a cropped region from one page of a technical document. "
    "Return the plain text representation of this region as if you were "
    "reading it naturally.\n"
    "Convert equations to LaTeX and tables to HTML.\n"
    "Do not speculate about content outside the crop."
)
```

Passe die Extract-Signatur-Docstring-Hilfe an (optional — Methode bleibt):

```python
    def extract(self, page_image: Any, page_number: int) -> ElementContent:
        self._load()
        text = self._run_ocr(page_image)
        return ElementContent(text=text)
```

(Der Parametername bleibt `page_image` für minimale Signaturänderung — semantisch ist es jetzt der Crop.)

- [ ] **Step 2: Regression-Check**

Run:
```bash
venv/bin/pytest extraction/tests/ -v
venv/bin/ruff check extraction
venv/bin/mypy extraction
```

Expected: alle PASS (keine Test-Änderung; nur Docstring + Prompt).

- [ ] **Step 3: Commit**

Run:
```bash
git add extraction/adapters/olmocr2_text.py
git commit -m "fix(extraction): olmocr2 erwartet Region-Crop, Prompt entsprechend angepasst"
```

---

## Task 12: `segment --out`-Default auf `cfg.output_dir`

**Files:**
- Modify: `extraction/__main__.py:30-33,54-55`
- Test: `extraction/tests/test_cli.py`

- [ ] **Step 1: Roten Test anhängen**

In `extraction/tests/test_cli.py`:

```python
def test_segment_uses_cfg_output_dir_when_out_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ohne --out greift cfg.output_dir."""
    calls: dict[str, object] = {}

    def _fake(pdfs: list[Path], cfg: object, output_base: Path) -> int:
        calls["out"] = output_base
        return 0

    monkeypatch.setattr("extraction.__main__.run_segment", _fake)

    from extraction.config import ExtractionConfig

    def _fake_cfg(_: object) -> ExtractionConfig:
        return ExtractionConfig(output_dir=str(tmp_path / "from_cfg"))

    monkeypatch.setattr("extraction.__main__._load_cfg", _fake_cfg)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    assert _invoke("segment", str(pdf)) == 0
    assert calls["out"] == tmp_path / "from_cfg"


def test_segment_out_overrides_cfg_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explizites --out hat Vorrang vor cfg.output_dir."""
    calls: dict[str, object] = {}

    def _fake(pdfs: list[Path], cfg: object, output_base: Path) -> int:
        calls["out"] = output_base
        return 0

    monkeypatch.setattr("extraction.__main__.run_segment", _fake)

    from extraction.config import ExtractionConfig

    def _fake_cfg(_: object) -> ExtractionConfig:
        return ExtractionConfig(output_dir=str(tmp_path / "from_cfg"))

    monkeypatch.setattr("extraction.__main__._load_cfg", _fake_cfg)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    assert _invoke("segment", str(pdf), "--out", str(tmp_path / "override")) == 0
    assert calls["out"] == tmp_path / "override"
```

- [ ] **Step 2: Tests laufen, müssen FAIL**

Run:
```bash
venv/bin/pytest extraction/tests/test_cli.py -v
```

Expected: `test_segment_uses_cfg_output_dir_when_out_missing` FAILS — aktueller Default ist hart `Path("outputs")`.

- [ ] **Step 3: Argparse-Default auf None, Resolution im Handler**

In `extraction/__main__.py`, ersetze die `segment`-Subparser-Zeile (Zeile 33):

```python
    seg.add_argument("--out", type=Path, default=None)
```

Ersetze den `segment`-Dispatch in `main()` (Zeile 54-55):

```python
    if args.command == "segment":
        out_base = args.out if args.out is not None else Path(cfg.output_dir)
        sys.exit(run_segment(args.pdfs, cfg, out_base))
```

Entferne den jetzt ungenutzten `DEFAULT_OUTPUT_BASE`-Import, falls er sonst nirgends verwendet wird:

```bash
grep -n DEFAULT_OUTPUT_BASE extraction/
```

Wenn nur in `__main__.py` → Import dort entfernen. Wenn auch in `config.py` weiterhin re-exportiert, Konstante belassen.

- [ ] **Step 4: Tests laufen, müssen PASS**

Run:
```bash
venv/bin/pytest extraction/tests/test_cli.py -v
```

Expected: alle PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add extraction/__main__.py extraction/tests/test_cli.py
git commit -m "fix(extraction): segment --out fällt auf cfg.output_dir zurück, statt hartem Default"
```

---

## Task 13: Doku-Drift bereinigen

**Files:**
- Modify: `.claude/CLAUDE.md:35`
- Modify: `docs/extraction_output.md:3,55,146-151`
- Modify: `docs/architecture.md:189,456`

- [ ] **Step 1: `.claude/CLAUDE.md`**

Ersetze den Zeilenblock um Zeile 35 (Commands-Abschnitt). Aktuell:
```
python -m extraction extract path/to/document.pdf --config config.yaml --output outputs/
```
Neuer Text:
```
# Extract eine PDF (4-stufiger Workflow)
python -m extraction segment path/to/document.pdf --config config.yaml
python -m extraction extract-text outputs/document
python -m extraction describe-figures outputs/document
python -m extraction assemble outputs/document
```

- [ ] **Step 2: `docs/extraction_output.md` — Einleitungssatz (Zeile 3)**

Ersetze:
```
Stabile Spezifikation des Outputs, den `python -m extraction extract <pdf>` produziert.
```
durch:
```
Stabile Spezifikation des Outputs, den der 4-stufige Workflow (`python -m extraction segment|extract-text|describe-figures|assemble`) produziert.
```

- [ ] **Step 3: `docs/extraction_output.md` — `rebuild`-Verweis (Zeile 55)**

Ersetze den Block:
```
python -m extraction rebuild outputs/<run>/
```
durch:
```
python -m extraction assemble outputs/<run>/
```

- [ ] **Step 4: `docs/extraction_output.md` — `segmentation.json`-Struktur (Zeile 146-151)**

Ersetze den Block:
````markdown
```jsonc
[
  {"page": 0, "bbox": [x0, y0, x1, y1], "region_type": "heading", "confidence": 0.99},
  {"page": 0, "bbox": [x0, y0, x1, y1], "region_type": "text",    "confidence": 0.95},
  {"page": 1, "bbox": [x0, y0, x1, y1], "region_type": "table",   "confidence": 0.93, "content": {"markdown": "..."}}
]
```
````
durch:
````markdown
```jsonc
{
  "doc_id": "sha256-16-hex",
  "source_file": "druckbericht.pdf",
  "total_pages": 3,
  "segmentation_tool": "mineru25",
  "regions": [
    {"page": 0, "bbox": [x0, y0, x1, y1], "region_type": "heading", "confidence": 0.99, "reading_order_index": 0},
    {"page": 0, "bbox": [x0, y0, x1, y1], "region_type": "text",    "confidence": 0.95, "reading_order_index": 1},
    {"page": 1, "bbox": [x0, y0, x1, y1], "region_type": "table",   "confidence": 0.93, "reading_order_index": 2, "content": {"markdown": "..."}}
  ]
}
```
````

- [ ] **Step 5: `docs/architecture.md` — zwei Stellen**

Suche zuerst:
```bash
grep -n "python -m extraction" docs/architecture.md
```

Ersetze in Zeile 189 (`python -m extraction rebuild outputs/<run>/`) durch:
```
python -m extraction assemble outputs/<run>/
```

Ersetze in Zeile 456 (`python -m extraction extract docs/fixtures/sample.pdf …`) durch den 4-stufigen Workflow:
```
python -m extraction segment docs/fixtures/sample.pdf --config config.yaml
python -m extraction extract-text outputs/sample
python -m extraction describe-figures outputs/sample
python -m extraction assemble outputs/sample
```

- [ ] **Step 6: Regression-Check — keine `python -m extraction extract ` mehr in Live-Docs**

Run:
```bash
grep -rn "python -m extraction extract\b" README.md docs/ .claude/CLAUDE.md
```

Expected: keine Treffer (historische Plans unter `docs/superpowers/plans/` und `docs/superpowers/specs/` sind Archiv, bleiben unangetastet).

- [ ] **Step 7: Commit**

Run:
```bash
git add .claude/CLAUDE.md docs/extraction_output.md docs/architecture.md
git commit -m "docs(extraction): Doku-Drift bereinigen (4-stufiger Workflow, segmentation.json-Struktur)"
```

---

## Task 14: Full-Suite Quality Gates

**Files:** keine Änderung, nur Verifikation.

- [ ] **Step 1: Pytest über alle Extraction-Tests**

Run:
```bash
venv/bin/pytest extraction/tests/ -v
```

Expected: alle PASS, keine Skips außer für GPU-gebundene Integration-Tests (`test_integration_mineru.py` kann ohne GPU skippen — das ist OK, aber dokumentieren).

- [ ] **Step 2: Ruff**

Run:
```bash
venv/bin/ruff check extraction
```

Expected: 0 Fehler.

Falls Fehler: fixen, nicht suppressen. Commit als separater `style(extraction): …`-Commit.

- [ ] **Step 3: Mypy**

Run:
```bash
venv/bin/mypy extraction
```

Expected: 0 Fehler. Typ-Annotationen in neuen Test-Helfern ergänzen, falls nötig (Striktheit bleibt).

- [ ] **Step 4: Zwischenstand committen, falls Fixes nötig waren**

```bash
git status
# Wenn unstaged Änderungen aus Ruff/Mypy-Fixes: committen als separaten Commit.
```

---

## Task 15: Merge und Push

**Files:** keine Änderung.

- [ ] **Step 1: Branch-State prüfen**

Run:
```bash
git log master..HEAD --oneline
git status
```

Expected: mehrere Commits auf der Branch, working tree clean.

- [ ] **Step 2: Master auf den neuesten Stand ziehen**

Run:
```bash
git fetch origin
git checkout master
git pull --ff-only origin master
```

Expected: master ist auf dem letzten Remote-Stand, fast-forward erfolgreich.

Falls der `pull` nicht fast-forwarden kann: STOP, mit dem User klären (es gibt divergente Änderungen auf remote).

- [ ] **Step 3: Zurück auf Branch, falls nötig rebase**

Run:
```bash
git checkout extraction-hardening-schema-stable
git rebase master
```

Expected: entweder „already up to date" oder sauberer Rebase. Falls Konflikte: STOP, mit dem User klären.

- [ ] **Step 4: Merge in master**

Run:
```bash
git checkout master
git merge --no-ff extraction-hardening-schema-stable -m "merge: extraction-hardening-schema-stable"
```

Expected: merge-commit entsteht, Fast-Forward vermieden, damit die Branch-Historie in master sichtbar bleibt.

- [ ] **Step 5: Final Quality Gates auf master**

Run:
```bash
venv/bin/pytest extraction/tests/ -q
venv/bin/ruff check extraction
venv/bin/mypy extraction
```

Expected: alle grün.

- [ ] **Step 6: Push**

Run:
```bash
git push origin master
git push origin extraction-hardening-schema-stable
```

Expected: beide Pushes erfolgreich. Branch bleibt auf remote zum späteren Aufräumen.

---

## Abschluss-Checks

Nach Task 15 verifizieren, dass die Spec-Ziele erfüllt sind:

- [ ] `extraction/stages/extract_text.py:_TARGET_TYPES` enthält TABLE und FORMULA.
- [ ] `extraction/interfaces.py` Docstrings beschreiben Crop-Semantik für `TextExtractor`.
- [ ] `extraction/interfaces.py` `FigureDescriptor.describe` hat optionalen `caption`-Parameter.
- [ ] `extraction/adapters/qwen25vl_figure.py` konstruiert Prompt mit Caption-Grounding.
- [ ] `extraction/__main__.py` `--out`-Default ist `None`, Resolution im Handler.
- [ ] `.claude/CLAUDE.md`, `docs/extraction_output.md`, `docs/architecture.md` zeigen den 4-stufigen Workflow.
- [ ] `docs/extraction_output.md` zeigt die korrekte `segmentation.json`-Struktur mit `doc_id`/`regions`-Wrapper.
- [ ] `grep -rn "python -m extraction extract\b" README.md docs/ .claude/CLAUDE.md` liefert keine Treffer.
- [ ] Keine Änderungen an `Element`, `ElementContent`, `content_list.json`-Schema oder Sidecar-Format.
