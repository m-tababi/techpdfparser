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

### Acceptance Criteria
- [ ] At least 4 tools are included in the comparison
- [ ] Comparison covers text, table, and formula extraction quality
- [ ] Results are documented in a structured comparison table (tool × criterion)
- [ ] A recommendation is written based on the findings
