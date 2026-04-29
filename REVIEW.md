# REVIEW.md

Konfiguration für den automatischen PR-Reviewer dieses Repos.
Overrides default review behavior. By default, review focuses on
correctness over style — this file extends and refines that.

Projekt-Kontext: `CLAUDE.md` (Architecture Invariants, Conventions),
`PRINCIPLES.md` (Team-Regeln). Verweisen, nicht wiederholen.

## Severity Calibration

🔴 **Important** (block merge):
- Architecture-Invariant verletzt (CLAUDE.md "Architecture Invariants"):
  3-Wege-Adapter-Alignment kaputt (Inv. 2), bbox nicht in PDF-Points
  (Inv. 3), Schema-Drift `extraction/models.py` ↔
  `docs/extraction_output.md` (Inv. 7), Stage-Marker-Logik falsch
  (Inv. 5), `segmentation.json.render_dpi` ignoriert (Inv. 6),
  `mypy`-Scope still erweitert (Inv. 8)
- Heavy dep auf Modul-Ebene importiert — bricht CPU-only
  (`docs/writing_adapters.md`)
- ML-Segmenter hardcoded `confidence = 1.0` (`docs/writing_adapters.md`)
- Extractor liefert Layout-Felder wie `caption` (gehört dem Segmenter
  laut CLAUDE.md Conventions)
- PRINCIPLES.md verletzt: Drive-by-Refactor in unrelated File (Scope),
  symptomatischer Patch statt Root Cause, Erfolg ohne Verification

🟡 **Nit** (worth fixing, not blocking):
- Neuer Code-Pfad ohne Test (bzw. ohne `@pytest.mark.integration`-Gate)
- `README.md` / `docs/architecture.md` Adapter-Tabelle nicht
  aktualisiert
- Adapter-Options am Top-Level `ExtractionConfig` statt unter
  `adapters: {<name>: {...}}`
- Kommentar wiederholt was der Code sagt (PRINCIPLES.md Communication)

🟣 **Pre-existing** (don't fix in this PR):
- Bugs in nicht-modifiziertem Code
- Phase-2-Features per `docs/extraction_output.md` "Was Phase 2 bringt":
  `document_rich.json`, populiertes `section_path`, `mentions`,
  `relations`, `caption_ref`
- Offene `backlog.md`-Items, die der PR nicht adressiert

## Always Check

- Neuer Adapter: Decorator-Name == `tool_name`-Property == Import in
  `extraction/adapters/__init__.py` (mit `try/except ImportError`)
- bbox: Pixel→Points im Adapter; Points→Pixel nur in
  `OutputWriter.crop_region` (`scale = dpi / 72`)
- Edit in `extraction/models.py` ↔ `docs/extraction_output.md`
  konsistent
- `.stages/<stage>.done|error` korrekt geschrieben/geräumt;
  `extract-text|describe-figures` Success räumt `assemble.done`
- Spätere Stages croppen mit `render_dpi` aus `segmentation.json`,
  nicht aus aktueller `cfg.dpi`
- Heavy deps lazy-geladen im Adapter-`_load()`, nicht auf Modul-Ebene
- `pyproject.toml`: `mypy`-Scope weiter `extraction/` only; neue
  Heavy deps in `[gpu]` oder `[tables]` Extras
- PRINCIPLES.md-Compliance: Scope Discipline, Root Causes,
  Verification Before Done

## Skip in Review

- Style / Formatting / Import-Order — `ruff` handled
- Typfehler außerhalb `extraction/` — out of `mypy`-Scope
- Files unter `outputs/` — gitignored Test-Artefakte
- `*.pdf` im Repo-Root — gitignored Test-Inputs
- Whitespace-only Diffs
- Pre-existing Bugs in nicht-modifiziertem Code
- Phase-2-Features (`document_rich.json`, `mentions`, `relations`)
- Triviale PRs (<10 geänderte Zeilen, einzelne Bugfixes): nur
  🔴-Findings posten, sonst Confirm-Kommentar

## Review Focus by Path

- `extraction/interfaces.py` — Protocol-breaking changes; jeder
  Adapter muss sein Protocol weiter erfüllen
- `extraction/registry.py` — selten editiert; sehr genau prüfen
- `extraction/adapters/__init__.py` — `try/except ImportError` um
  jeden heavy-deps-Adapter-Import erhalten
- `extraction/adapters/<new>.py` — 3-Wege-Alignment, lazy `_load()`,
  echte ML-Confidence (1.0 nur mit Rule-based-Kommentar)
- `extraction/models.py` — jeder Edit reconcile mit
  `docs/extraction_output.md`
- `extraction/output.py` — einzige bbox-Scaling-Site; hohe Blast-Radius
- `extraction/stages/*.py` — Stage-Idempotenz, Marker-Writes,
  `render_dpi`-aus-segmentation Invariant
- `extraction/config.py` — keine adapter-spezifischen Keys top-level
- `configs/*.yaml` — Adapter-Options unter `adapters:`, nicht im
  top-level `extraction:`
