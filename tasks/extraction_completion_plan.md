# Plan: Extraktion + Speicherung auf Projekt.md-Niveau bringen

> **Status: Vollständig umgesetzt.** Dieser Plan ist archiviert.
> Offene Punkte für Phase 2 am Ende.

---

## Was umgesetzt wurde

| # | Feature | Datei | Status |
|---|---|---|---|
| 1 | Strukturerkennung (TOC + Font-Heuristik) | `src/utils/sections.py` | ✅ fertig |
| 2 | `TextChunk` mit `section_title`, `section_path`, `heading_level` | `src/core/models/elements.py` | ✅ fertig |
| 3 | Section-aware Chunker | `src/adapters/chunkers/section_aware.py` | ✅ fertig |
| 4 | `Formula.latex` Default `""` (kein Crash mehr) | `src/core/models/elements.py` | ✅ fertig |
| 5 | `Figure.caption` via Heuristik befüllt | `src/adapters/parsers/pdfplumber_parser.py` | ✅ fertig |
| 6 | `pymupdf_structured` Extractor | `src/adapters/ocr/pymupdf_structured.py` | ✅ fertig |
| 7 | `sections.json` in Text-Run gespeichert | `src/core/pipelines/text.py` | ✅ fertig |
| 8 | Structured-Pipeline linkt Sections via `sections.json` | `src/core/pipelines/structured.py` | ✅ fertig |
| 9 | `storage.latest_text_sections()` | `src/utils/storage.py` | ✅ fertig |
| 10 | Config-Einträge für neue Adapter | `config.example.yaml`, `config.yaml` | ✅ fertig |

---

## Offene Punkte (Phase 2)

- **Unified Retrieval API**: `UnifiedRetriever` als einzelner Query-Entrypoint (alle 5 Collections parallel)
- **Score-Normalisierung**: RRF-Scores auf [0, 1] normieren für einheitliche Vergleichbarkeit
- **CLI `query`-Subcommand**: `python -m src query "Frage hier"` analog zu `ingest`
- **BenchmarkRunner** (Phase 3): A/B-Vergleich zwischen Adapter-Kombinationen
