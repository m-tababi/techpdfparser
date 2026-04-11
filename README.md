# techpdfparser

A local-first, multi-pipeline system for extracting and indexing technical PDFs into a vector database. Three independent pipelines handle visual pages, text chunks, and structured elements (tables, formulas, figures). Every tool is swappable via config — no code changes needed.

## Architecture

```
PDF
 ├── Visual Pipeline     PyMuPDF → ColQwen2.5 (multi-vector)    → Qdrant: visual_pages
 ├── Text Pipeline       olmOCR2 → FixedSizeChunker → BGE-M3    → Qdrant: text_chunks
 └── Structured Pipeline MinerU2.5 → PP-FormulaNet / Qwen2.5-VL → Qdrant: tables, formulas, figures
```

At query time, `UnifiedRetriever` hits all five collections in parallel and fuses results via Reciprocal Rank Fusion (RRF).

### Adapter registry

Every adapter self-registers on import:

```python
@register_renderer("pymupdf")
class PyMuPDFRenderer: ...
```

Pipelines look up by name: `get_renderer("pymupdf", **kwargs)`. Swapping a tool means changing one string in `config.yaml` — never touching existing adapters.

### Dependency direction

```
utils → core/models → core/interfaces → core/pipelines → adapters
```

No upward imports. GPU models are lazy-loaded on first use.

## Project Structure

```
src/
  core/
    config.py          # YAML → Pydantic AppConfig
    registry.py        # Adapter registry & lookup
    retrieval.py       # UnifiedRetriever (query all collections → RRF)
    models/            # DocumentMeta, ExtractedElement subtypes, RetrievalResult
    interfaces/        # Protocol definitions for every component
    pipelines/         # VisualPipeline, TextPipeline, StructuredPipeline
  adapters/
    renderers/         # pymupdf
    visual/            # colqwen25, clip
    ocr/               # olmocr2, pymupdf_text
    chunkers/          # fixed_size
    embedders/         # bge_m3, minilm
    parsers/           # mineru25, pdfplumber
    formula/           # ppformulanet, pix2tex
    figures/           # qwen25vl, moondream, noop
    fusion/            # rrf, score_norm
    vectordb/          # qdrant
  utils/
    ids.py             # Stable SHA256-based point IDs
    storage.py         # Versioned output dirs
    jsonl.py           # JSONL read/write helpers
    manifest.py        # Per-run manifest files
    timing.py          # Timing context manager
    logging.py         # Structured logging
tasks/
  todo.md              # Roadmap with checkable items
  lessons.md           # Post-mortems and lessons learned
tests/
config.example.yaml    # Annotated reference config
```

## Setup

### Prerequisites

- Python 3.10+
- Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`)

### NVIDIA GPU (CUDA)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -r requirements-gpu.txt
```

### AMD GPU / CPU (no CUDA)

Uses lighter fallback adapters: `pymupdf_text` instead of olmOCR2, `minilm` instead of BGE-M3, `clip` instead of ColQwen2.5, `pdfplumber` instead of MinerU, `pix2tex` (CPU ViT) for formulas, `moondream` (~2B, CPU-viable) for figures.

```bash
python -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-amd.txt
```

### Dev / testing (no GPU)

```bash
pip install -r requirements-dev.txt
```

## Configuration

```bash
cp config.example.yaml config.yaml
# Edit config.yaml — set Qdrant host, device, model paths, etc.
```

Key config blocks:

```yaml
pipelines:
  visual:
    renderer: "pymupdf"     # swap to any registered renderer
    embedder: "colqwen25"   # AMD: "clip"
  text:
    extractor: "olmocr2"    # AMD: "pymupdf_text"
    embedder: "bge_m3"      # AMD: "minilm"
  structured:
    parser: "mineru25"      # AMD: "pdfplumber"
    formula_extractor: "ppformulanet"   # AMD: "pix2tex"
    figure_descriptor: "qwen25vl"       # AMD: "moondream"

adapters:
  qdrant:
    host: "localhost"
    port: 6333
```

See `config.example.yaml` for all options with inline comments.

## Usage

### Ingest a PDF

```bash
python -m techpdfparser ingest path/to/document.pdf
python -m techpdfparser ingest path/to/document.pdf --config config.yaml
```

Output:

```
Ingesting document.pdf | doc_id=a3f1b2c4d5e6f7a8 | pages=42
  Visual:     42 pages indexed
  Text:       187 chunks indexed
  Structured: 14 tables, 6 formulas, 9 figures indexed

  Outputs: outputs/a3f1b2c4d5e6f7a8/
```

### Query (Python API)

```python
from src.core.retrieval import UnifiedRetriever

retriever = UnifiedRetriever(
    retrieval_engine=qdrant_engine,
    visual_embedder=colqwen_embedder,
    text_embedder=bge_embedder,
    fusion_engine=rrf_engine,
    visual_collection="visual_pages",
    text_collection="text_chunks",
    tables_collection="tables",
    formulas_collection="formulas",
    figures_collection="figures",
)

results = retriever.query("heat dissipation in multilayer PCBs", top_k=10)
```

## Development

```bash
# Run all tests
pytest

# Single file
pytest tests/test_chunker.py

# Single test
pytest tests/test_rrf.py::test_rrf_basic

# Lint
ruff check src tests

# Type check
mypy src
```

## ID Stability

Point IDs are deterministic: SHA256 of `doc_id:page:type:tool:seq` truncated to 16 hex chars. Re-ingesting the same PDF with the same config upserts identical IDs — safe for deduplication and incremental re-runs.

## Current Status

| Phase | Status |
|---|---|
| Phase 1 — skeleton, all interfaces, all adapters, Qdrant, RRF, 48 tests | Complete |
| Phase 2 — unified retrieval API, score normalization, CLI entrypoint | In progress |
| Phase 3 — BenchmarkRunner, A/B comparison reports | Planned |
