---

## 2026-04-16 — Compare PDF Extraction Tools (OCR Benchmark)

**Priority:** Important

### User Story
As a developer, I want to compare different PDF extraction tools across key quality dimensions so that I can make an informed decision about which tool to use in the extraction pipeline.

### Tasks
- [ ] Define comparison criteria (text accuracy, table extraction, formula handling, layout preservation, speed, local-first support)
- [ ] Select tools to benchmark (e.g. MinerU, PyMuPDF, Docling, OlmOCR, Unstructured.io)
- [ ] Prepare a representative test set of PDFs (cover technical docs, tables, formulas, multi-column layouts)
- [ ] Run extraction with each tool on the test set and collect outputs
- [ ] Evaluate outputs against criteria and document results in a comparison table
- [ ] **Per-role extractor benchmark.** Run the extraction pipeline with
      different `table_extractor`, `formula_extractor`, `figure_descriptor`
      combinations on the corpus. Produce a per-role recommendation table.
      Unblocks data-driven default changes in `ExtractionConfig`. Cross-ref
      target for `docs/extraction_output.md` → Merge-Regeln.

### Acceptance Criteria
- [ ] At least 4 tools are included in the comparison
- [ ] Comparison covers text, table, and formula extraction quality
- [ ] Results are documented in a structured comparison table (tool × criterion)
- [ ] A recommendation is written based on the findings

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
