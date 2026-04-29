# CLAUDE.md

Project-specific guidance for `techpdfparser` (Phase 1 of
BAM33FE2502_KI-Dokumentenanalyse). The global rules in `~/.claude/CLAUDE.md`
apply too — this file only adds what is project-specific.

## Overview

`techpdfparser` is the **extraction layer** (Layer 1) of a multi-stage system
that will verify technical claims in PDFs and trace them back to evidence —
ultimately to support approval of safety-critical containers. The block
produces a stable `content_list.json` contract; downstream layers depend on
the contract, not adapter internals. Tools are swappable per role via YAML
config; heavy work runs in four staged subcommands so each model releases GPU.

See @docs/architecture.md for the full picture.

## Where to Look

- **Role interfaces (Protocols):** `extraction/interfaces.py`
- **Adapter registry:** `extraction/registry.py`
- **Adapter list + lazy imports:** `extraction/adapters/__init__.py`
- **Config model + defaults:** `extraction/config.py`
- **Pydantic schema:** `extraction/models.py`
- **Output writer + crop scaling:** `extraction/output.py`
- **Stages:** `extraction/stages/{segment,extract_text,describe_figures,assemble}.py`
- **CLI entrypoint:** `extraction/__main__.py`
- **Output format spec:** @docs/extraction_output.md
- **Adapter checklist:** @docs/writing_adapters.md
- **Architecture overview:** @docs/architecture.md
- **Quality-gate scope (mypy/ruff/pytest):** @pyproject.toml

## Commands

Manual venv + pip (no uv); see @README.md for install.

```bash
python -m extraction segment <pdf>...
python -m extraction extract-text <outdir>...
python -m extraction describe-figures <outdir>...
python -m extraction assemble <outdir>...
pytest -q                    # default: integration tests are gated out
pytest -m integration        # GPU/MinerU/model downloads required
ruff check extraction
mypy
```

## Architecture Invariants

1. **Six swappable roles** are defined as `Protocol`s in
   `extraction/interfaces.py`: `PageRenderer`, `Segmenter`, `TextExtractor`,
   `TableExtractor`, `FormulaExtractor`, `FigureDescriptor`. Every concrete
   adapter implements exactly one.
2. **Adapter registration — three things must align:**
   - Class is decorated with `@register_<role>("<name>")` from
     `extraction/registry.py`.
   - Module is imported in `extraction/adapters/__init__.py`, wrapped in
     `try/except ImportError` if it pulls heavy deps (Torch / Transformers /
     MinerU). The CLI imports the package, which triggers registration.
   - `tool_name` property == decorator name. The pipeline matches segmenter
     vs extractor `tool_name` to skip passthrough work; a mismatch silently
     breaks that optimization.
3. **bbox is in PDF-points, top-left origin.** Any pixel→points conversion
   lives in the adapter; the only points→pixel scaling site is
   `OutputWriter.crop_region` (`scale = dpi / 72`).
4. **Per-element sidecars (`pages/<n>/<el_id>_<type>.json`) are the source
   of truth.** `assemble` rebuilds `content_list.json` deterministically,
   sorted by `(page, reading_order_index, element_id)` and globally re-numbered.
5. **Stage state is filesystem-backed.** `.stages/<stage>.done` and
   `.stages/<stage>.error` markers gate re-runs; `segment` validates
   PDF-hash + source + render_dpi + stage-config against `segmentation.json`.
6. **`segmentation.json.render_dpi` wins over current config.** Later stages
   crop using the stored DPI; config changes don't silently misalign crops.
7. **Schema lives in `extraction/models.py`** and must stay consistent with
   @docs/extraction_output.md. If they diverge, code wins; propose a doc
   edit.
8. **mypy scope is `extraction/` only** per @pyproject.toml. Other
   top-level dirs (`embedding/`, `indexing/`, `scripts/`) are not type-gated.

## Conventions

- **New adapter:** new file `extraction/adapters/<name>.py`, decorate the
  class with `@register_<role>("<name>")`, expose `tool_name`, lazy-load
  heavy deps inside a `_load()` helper (not at import). Add a `try/except`
  import line in `extraction/adapters/__init__.py`. See
  @docs/writing_adapters.md.
- **Adapter-specific options** go under `adapters: { <name>: {...} }` in
  YAML and are read via `cfg.get_adapter_config(name)`. Don't add per-adapter
  keys to the top-level `ExtractionConfig`.
- **Tests:** `extraction/tests/test_<module>.py`. Heavy/GPU/model-download
  tests use `@pytest.mark.integration` so the default `pytest -q` stays fast.
- **End-to-end configs** live in `configs/` (one YAML per setup);
  **outputs** go under `outputs/<pdf-stem>/`.
- **Extractor adapters return content fields only**
  (`text`/`markdown`/`latex`/`description`). Layout fields (`caption`)
  belong to the segmenter; the merge happens in stage code, not in extractors.
- **Segmenter confidence:** ML adapters propagate the model's per-region
  score into `Region.confidence`. Rule-based segmenters may use `1.0` but
  must comment why.
