# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run a single test file
pytest tests/test_chunker.py

# Run a single test
pytest tests/test_rrf.py::test_rrf_basic

# Lint
ruff check src tests

# Type check
mypy src
```

## Architecture

Three independent pipelines process PDFs into Qdrant collections:

| Pipeline | Flow | Collections |
|---|---|---|
| Visual | PyMuPDF → ColQwen2.5 → Qdrant | `visual_pages` |
| Text | olmOCR2 → FixedSizeChunker → BGE-M3 → Qdrant | `text_chunks` |
| Structured | MinerU2.5 → PP-FormulaNet / Qwen2.5-VL → BGE-M3 → Qdrant | `tables`, `formulas`, `figures` |

**Dependency direction** (no upward imports): `utils` → `core/models` → `core/interfaces` → `core/pipelines` → `adapters`

### Adapter registry (`src/core/registry.py`)

Tool switching is config-driven. Every adapter self-registers on import:

```python
@register_renderer("pymupdf")
class PyMuPDFRenderer: ...
```

Pipelines look up by name: `get_renderer("pymupdf", **kwargs)`. To add a new tool: create a new file in the appropriate `adapters/` subdirectory, add the decorator, and register nothing else — never modify existing adapters.

### Config system

`config.example.yaml` → copy to `config.yaml`. The `adapters:` block holds per-adapter kwargs; adapter names in `pipelines:` select which class to instantiate. Swapping a tool means changing one string in `config.yaml`.

### ID generation (`src/utils/ids.py`)

Stable, deterministic IDs: SHA256 of `doc_id:page:type:tool:seq` truncated to 16 hex chars. Used as Qdrant point IDs — must remain stable across re-runs for deduplication.

## Coding rules

- Functions: one job, ~20–30 lines before splitting
- Files: ≤ 200–300 lines
- No abstractions unless immediately needed (YAGNI)
- Comments explain *why*, not what
- No dead code — delete it or add a `TODO:` with reason
- No global state; no circular imports

## Current state

- Phase 1 complete: all interfaces, adapters (GPU models lazy-loaded), Qdrant writer/retriever, RRF fusion, 48 tests passing
- Phase 2 in progress: unified retrieval API, CLI entrypoint (`python -m techpdfparser ingest <pdf>`), score normalization, figure enrichment
- Phase 3 planned: `BenchmarkRunner`, A/B comparison reports

Track work in `tasks/todo.md`; record post-mortems in `tasks/lessons.md`.
