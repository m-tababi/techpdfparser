> **ARCHIVED — 2026-04-18.** References the pre-refactor `src/…` layout.
> Current extraction lives under `extraction/`. Kept for history only.
> See `docs/extraction_output.md` and
> `docs/superpowers/specs/2026-04-18-extraction-hardening-design.md`.

# Todo — Multi-Pipeline PDF Intelligence System

## Phase 1 — Architecture Skeleton (current)

### Core Foundation
- [x] Data models: `DocumentMeta`, `BoundingBox`, `ExtractedElement` subtypes, `RetrievalResult`
- [x] Protocol interfaces: `PageRenderer`, `VisualEmbedder`, `TextExtractor`, `TextChunker`, `TextEmbedder`, `StructuredParser`, `FormulaExtractor`, `FigureDescriptor`, `IndexWriter`, `RetrievalEngine`, `FusionEngine`, `BenchmarkRunner`
- [x] Central config system (YAML → Pydantic)
- [x] Adapter registry (name → class, config-driven tool switching)
- [x] Utils: ID generation, timing, storage paths, logging

### Pipelines
- [x] `VisualPipeline`: render → embed → index
- [x] `TextPipeline`: extract → chunk → embed → index
- [x] `StructuredPipeline`: parse → enrich (formula/figure) → embed → index

### Adapters (Phase 1)
- [x] `PyMuPDFRenderer` — page rendering
- [x] `ColQwen25Embedder` — visual multi-vector embedding
- [x] `OlmOCR2Extractor` — OCR + reading-order text extraction
- [x] `FixedSizeChunker` — baseline text chunking
- [x] `BGEM3Embedder` — text embedding
- [x] `MinerU25Parser` — structured element extraction
- [x] `PPFormulaNetExtractor` — formula recognition
- [x] `Qwen25VLDescriptor` — figure description
- [x] `QdrantIndexWriter` + `QdrantRetrievalEngine` — vector DB backend
- [x] `ReciprocalRankFusion` — result fusion

### Tests
- [x] `test_models.py` — Pydantic serialization/deserialization
- [x] `test_config.py` — YAML config loading
- [x] `test_ids.py` — ID generation stability
- [x] `test_chunker.py` — fixed-size chunking logic
- [x] `test_rrf.py` — RRF fusion correctness
- [x] `test_pipelines.py` — pipeline orchestration with mocked adapters

---

## Phase 2 — Retrieval API + Fusion (next)
- [ ] Unified retrieval API (query → all collections → fusion)
- [ ] Score normalization across collections
- [ ] Figure description enrichment (load image → call VLM)
- [ ] Formula image rendering for PP-FormulaNet standalone flow
- [ ] CLI entrypoint: `python -m techpdfparser ingest <pdf>`

## Phase 3 — Benchmarking (later)
- [ ] `BenchmarkRunner` implementation
- [ ] Latency + memory + storage tracking per pipeline run
- [ ] Output comparator (diff two runs side-by-side)
- [ ] A/B comparison reports

---

## Review
<!-- Add post-implementation notes here -->
