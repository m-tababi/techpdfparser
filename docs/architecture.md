# Architecture Overview

> Companion document for newcomers. Explains the idea behind `techpdfparser`,
> the current staged extraction architecture, and how to add a new adapter.
>
> For the output format spec, see `docs/extraction_output.md`.
> For the adapter-author checklist, see `docs/writing_adapters.md`.
> Code is authoritative. Where this document and code disagree, trust the code.

---

## 1. Vision

The long-term goal is a system that verifies technical claims in PDFs and traces
them back to supporting evidence: text, tables, formulas, figures, diagrams, and
technical drawings.

The system is split into independently replaceable layers:

```
Layer 4 — Agent / Routing
Layer 3 — Storage
Layer 2 — Embedding
Layer 1 — Extraction   <-- current repository scope
```

Layer 1 produces a stable `content_list.json` contract. Later layers should
depend on that contract, not on the extraction implementation details.

---

## 2. Current Extraction Flow

Extraction is intentionally staged. Each heavy model runs in its own process, so
GPU memory is released naturally between stages.

```
PDF
 │
 ▼
segment
  - render page images
  - run layout segmenter
  - write segmentation.json
  - write passthrough sidecars when segmenter content matches the configured role
 │
 ▼
extract-text
  - handle text, heading, table, and formula regions that still need sidecars
  - crop regions from saved page images
  - call configured text/table/formula extractors
 │
 ▼
describe-figures
  - handle figure, diagram, and technical_drawing regions
  - crop regions from saved page images
  - call configured figure descriptor
 │
 ▼
assemble
  - rebuild content_list.json deterministically from sidecars
```

CLI entrypoint: `extraction/__main__.py`

```
python -m extraction segment <pdf1> <pdf2> ... [--config extraction_config.yaml] [--out outputs]
python -m extraction extract-text <outdir1> <outdir2> ... [--config extraction_config.yaml] [--force]
python -m extraction describe-figures <outdir1> <outdir2> ... [--config extraction_config.yaml] [--force]
python -m extraction assemble <outdir1> <outdir2> ... [--config extraction_config.yaml]
```

If `--config` is omitted, the CLI loads `extraction_config.yaml` from the current
working directory if present; otherwise it uses `ExtractionConfig()` defaults.
If `segment --out` is omitted, `cfg.output_dir` is used.

---

## 3. Roles And Available Adapters

The pipeline depends on six swappable roles, defined as `Protocol`s in
`extraction/interfaces.py`. Adapters self-register under a registry name —
that string is what goes into YAML. The currently active selection lives in
`extraction/config.py`; switching tools is a YAML change, no code edit.

| Role | Config key | Registered adapter names |
| --- | --- | --- |
| `PageRenderer` | `renderer` | `pymupdf` |
| `Segmenter` | `segmenter` | `mineru25`, `mineru_hybrid`, `mineru_vlm`, `pymupdf_text` |
| `TextExtractor` | `text_extractor` | `mineru25`, `mineru_hybrid`, `mineru_vlm`, `olmocr2`, `noop` |
| `TableExtractor` | `table_extractor` | `mineru25`, `mineru_hybrid`, `mineru_vlm`, `qwen25vl_table`, `tatr`, `docling_table`, `noop` |
| `FormulaExtractor` | `formula_extractor` | `mineru25`, `mineru_hybrid`, `mineru_vlm`, `noop` |
| `FigureDescriptor` | `figure_descriptor` | `qwen25vl`, `noop` |

The three MinerU names all live in `extraction/adapters/mineru25_segmenter.py`
and differ only in which MinerU backend they invoke (`pipeline`,
`hybrid-auto-engine`, `vlm-auto-engine` — see README.md "MinerU-Backend wechseln").
As extractors they are **passthroughs**: when segmenter and extractor share
the same registry name, `segment` writes the segmenter's content directly as
a sidecar — no second model call (Tool-Match-Optimierung, siehe
@docs/extraction_output.md).

CPU-only adapters: `pymupdf` (renderer), `pymupdf_text` (segmenter),
`noop` (every extractor / descriptor role). A CPU-only run needs a config
that uses only these.

---

## 4. Config Dictates Behavior

`extraction/config.py` is the source of truth for the active selection. A YAML
config can override every top-level role name and provide adapter-specific
keyword arguments. Example structure (values picked from the table above):

```yaml
extraction:
  renderer: pymupdf
  segmenter: <segmenter_name>
  text_extractor: <text_extractor_name>
  table_extractor: <table_extractor_name>
  formula_extractor: <formula_extractor_name>
  figure_descriptor: <figure_descriptor_name>
  output_dir: outputs
  dpi: 150
  confidence_threshold: 0.3

adapters:
  <adapter_name>:
    # adapter-specific keyword arguments, e.g. device: cuda
```

The pipeline does not choose tools heuristically. It looks up exactly the role
adapter named in config. If a role tool has the same `tool_name` as the
segmenter and the segmenter already emitted content, the `segment` stage writes
that content as a sidecar. Later stages skip existing sidecars unless explicitly
forced or rerun without a done marker.

---

## 5. Output Contract

Every output run lives in one output directory, usually `outputs/<pdf-stem>/`.

Key files:

- `segmentation.json`: raw segmenter regions plus run metadata.
- `pages/<n>/page.png`: rendered page image.
- `pages/<n>/<element_id>_<type>.json`: per-element source-of-truth sidecar.
- `pages/<n>/<element_id>_<type>.png`: crop for visual element types.
- `content_list.json`: merged flat list created by `assemble`.
- `.stages/<stage>.done` and `.stages/<stage>.error`: stage state markers.

The per-element sidecars are the source of truth. `content_list.json` can be
rebuilt at any time with:

```
python -m extraction assemble outputs/<run>
```

For the complete schema, see `docs/extraction_output.md`.

---

## 6. Stage Safety Rules

Stage state is explicit and filesystem-backed:

- `segment` refuses to process a non-empty output directory when
  `.stages/segment.done` is missing. That usually means a prior crashed or
  interrupted run left orphaned artefacts.
- If `.stages/segment.done` exists, `segment` validates the existing
  `segmentation.json` against the current PDF hash, source filename, effective
  render DPI, and segment-stage config. Mismatches are written to
  `.stages/segment.error`.
- `extract-text`, `describe-figures`, and `assemble` require their predecessor
  marker files before running.
- `extract-text --force` and `describe-figures --force` delete the target
  sidecars/crops for their own stage before regenerating them.
- A successful `extract-text` or `describe-figures` clears `assemble.done`,
  because `content_list.json` may now be stale.
- Per-PDF/per-output failures are isolated: one failed item does not stop the
  rest of the stage batch.

`segment` stores `render_dpi` in `segmentation.json`. Later stages use that
stored value for cropping, so changing the current config DPI after segmentation
does not silently misalign crops.

---

## 7. Data Model

All schema models live in `extraction/models.py`.

```
Region
  page, bbox, region_type, reading_order_index, confidence, content?

Element
  element_id, type, page, bbox, reading_order_index,
  section_path, confidence, extractor, content

ElementContent
  text, markdown, html, latex, image_path, description, caption,
  caption_position, footnotes, markers

ContentList
  doc_id, source_file, total_pages, schema_version,
  segmentation_tool, pages, elements
```

`DocumentRich`, `Section`, and `Relation` are defined but not produced by the
current staged pipeline. `document_rich.json`, populated `section_path`, mentions,
and relations remain phase-2 work.

---

## 8. Adding A New Tool

Example: a hypothetical table extractor named `tabulite`.

1. Implement the relevant Protocol.

```python
from PIL.Image import Image

from ..models import ElementContent
from ..registry import register_table_extractor


@register_table_extractor("tabulite")
class TabuliteTableExtractor:
    TOOL_NAME = "tabulite"

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._engine = None

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def _load(self) -> None:
        if self._engine is not None:
            return
        import tabulite

        self._engine = tabulite.load(self._model_path)

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        self._load()
        result = self._engine.analyze(region_image)
        return ElementContent(
            markdown=result.markdown or None,
            text=result.plain_text or None,
        )
```

2. Register the module import in `extraction/adapters/__init__.py`.

```python
try:
    from . import tabulite_table  # noqa: F401
except ImportError as exc:
    log.debug("tabulite_table not registered: %s", exc)
```

3. Configure it.

```yaml
extraction:
  segmenter: mineru25
  table_extractor: tabulite

adapters:
  tabulite:
    model_path: /opt/models/tabulite-v2.bin
```

4. Test it.

Keep lightweight adapter tests in the default test suite. Mark real GPU/model
checks with `@pytest.mark.integration`.

```
pytest -q
ruff check extraction
mypy
```

5. Smoke-test with the staged CLI.

```
python -m extraction segment docs/fixtures/sample.pdf --config configs/with_tabulite.yaml
python -m extraction extract-text outputs/sample --config configs/with_tabulite.yaml
python -m extraction describe-figures outputs/sample --config configs/with_tabulite.yaml
python -m extraction assemble outputs/sample --config configs/with_tabulite.yaml
```

For adapter-specific rules, see `docs/writing_adapters.md`.

---

## 9. Where To Look

| Concern | File |
| --- | --- |
| CLI and config loading | `extraction/__main__.py` |
| Stage orchestration | `extraction/stages/` |
| Stage marker/reporting helpers | `extraction/stages/__init__.py` |
| Role interfaces | `extraction/interfaces.py` |
| Registry mechanics | `extraction/registry.py` |
| Config model and defaults | `extraction/config.py` |
| Data models | `extraction/models.py` |
| Output writer and crop scaling | `extraction/output.py` |
| Adapter list and lazy imports | `extraction/adapters/__init__.py` |
| Output format spec | `docs/extraction_output.md` |
| Adapter-author checklist | `docs/writing_adapters.md` |
| Open work items | `backlog.md` |

When in doubt, read the code. It is the current source of truth.
