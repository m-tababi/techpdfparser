# Design: Extraction-Härtung ohne Schema-Bruch

**Datum:** 2026-04-24
**Status:** draft — review pending
**Vorgänger:** `docs/superpowers/specs/2026-04-23-staged-extraction-pipeline-design.md`

## Warum

Nach dem Re-Run von `jmmp-09-00199-v2` sind fünf Lücken zwischen Code und Contract sichtbar. Vier davon sind latent (greifen erst bei nicht-default Configs) oder reine Doku-Drift. Punkt 3 (FigureDescriptor ohne Caption-Kontext) wirkt schon heute: im aktuellen Run produzieren Sub-Figures auf Seite 6 sichtbare Halluzinationen (siehe `b83b341b9ca31016`, `f7c30b7a776b2189`). Kein Schema-Bruch nötig, um sie zu beheben.

1. Stage 2 (`extract-text`) verarbeitet nur `TEXT`/`HEADING`. Tabellen und Formeln fallen bei role-mismatch still durch. Evidenz: `extraction/stages/extract_text.py:17` (`_TARGET_TYPES = {TEXT, HEADING}`).
2. `TextExtractor.extract()` bekommt Vollseite, nicht Region-Crop — Docstring und Call-Site. Evidenz: `extraction/interfaces.py:40`, `extraction/stages/extract_text.py:100-101`, `extraction/adapters/olmocr2_text.py:104-107`.
3. `FigureDescriptor.describe()` bekommt keinen Caption-Kontext → kontextfreie Halluzinationen bei Sub-Figures. Evidenz: `extraction/adapters/qwen25vl_figure.py:93-117`.
4. `segment --out` default überschreibt `cfg.output_dir`, silent config-ignore. Evidenz: `extraction/__main__.py:33`.
5. Doku-Drift: `.claude/CLAUDE.md:35`, `docs/extraction_output.md:3,146-151`, `docs/architecture.md:189,456` verweisen auf den entfernten `extract`-Command oder auf das falsche `segmentation.json`-Format.

## Scope

### 1. Stage-Verhalten sauberziehen

- Stage 1 (`segment`) bleibt unverändert: Rendering, `segmentation.json`, und **Passthrough-Sidecars nur dann**, wenn `role_tool == segmentation_tool` und `region.content` verwertbar ist. (`extraction/stages/segment.py:133-161` — ist bereits so.)
- Stage 2 (`extract-text`) erweitert `_TARGET_TYPES` auf `{TEXT, HEADING, TABLE, FORMULA}`. CLI-Name bleibt.
- Pro Region in Stage 2:
  - Wenn für die Region-ID bereits ein Sidecar im Page-Ordner liegt (Stage 1 hat ihn im Role-Match-Pfad geschrieben): **überspringen**, nichts neu extrahieren. In einem sauberen Lauf ist das äquivalent zum Role-Match-Check (`role_tool == segmentation_tool`).
  - Sonst: Region-Crop aus `pages/<N>/page.png` bauen, passenden Extractor aufrufen, Sidecar schreiben.
- Drop-/Persist-Regeln (konsistent mit `docs/extraction_output.md` § Merge-Regeln):
  - `text`, `heading`: Sidecar wird gedroppt, wenn `content.text` leer/whitespace ist.
  - `table`, `formula`: Sidecar persistiert, solange `image_path` gesetzt ist — auch wenn `markdown`/`latex`/`text` leer bleiben. Der Crop wird im Extraktionspfad immer geschrieben. Bei Tables wird `html` zusätzlich eingetragen, wenn vorhanden.
- `caption` aus dem Segmenter bleibt in **jedem** Extraktionspfad erhalten (Muster: `extraction/stages/extract_text.py:102-103`; gleiche Zuweisung in den neuen Table/Formula-Zweigen einsetzen).
- Die bestehenden `MinerU25{Text,Table,Formula}Extractor`-Passthroughs (`extraction/adapters/mineru25_segmenter.py:109-173`) bleiben im Registry. Sie werden nach dem Refactor vom Stage-2-Code im Role-Match-Pfad **nicht mehr aufgerufen** — der Skip passiert über die Sidecar-Existenz, vor dem Extractor-Lookup.

### 2. Interface-Anpassungen

- `TextExtractor.extract(image, page_number)`: Signatur unverändert, aber `image` ist semantisch **Region-Crop**. Docstring (`extraction/interfaces.py:40`) und `olmocr2_text.py` auf Crop umstellen; der `_TARGET_LONGEST_DIM`-Resize bleibt erhalten, wirkt dann auf den Crop.
- `FigureDescriptor.describe(image, caption=None)`: neuer optionaler `caption`-Parameter, Default `None`. Bestehende einfache Adapter können ihn ignorieren. `qwen25vl_figure.py` nutzt ihn im Prompt.
- **Kein** Schema-Change an `Element`, `ElementContent`, `segmentation.json` oder `content_list.json`.
- **Kein** neuer CLI-Step.

### 3. Contracts und Docs synchronisieren

- `segment --out`: wenn weggelassen, greift `cfg.output_dir`. Implementierung: argparse-Default ist `None`, Resolution im Handler (`args.out or Path(cfg.output_dir)`).
- `docs/extraction_output.md:146-151`: `segmentation.json`-Struktur an den echten Code (`{doc_id, source_file, total_pages, segmentation_tool, regions: [...]}`) anpassen.
- Veraltete `python -m extraction extract ...`-Verweise bereinigen:
  - `.claude/CLAUDE.md:35`
  - `docs/extraction_output.md:3`
  - `docs/architecture.md:189,456`
- `docs/extraction_output.md:55` (Verweis auf `python -m extraction rebuild`) entfernen — `assemble` übernimmt diese Rolle; kein neuer `rebuild`-Command.

### 4. Figure-Qualität gezielt härten

- Qwen-Prompt nimmt die Caption als zusätzlichen Kontext, wenn gesetzt. Instruktion: sichtbaren Inhalt beschreiben, Caption als Stütze nutzen, **nichts erfinden**, was weder Crop noch Caption belegt.
- Bei Caption `None` oder leer: Prompt fällt auf Bild-only zurück (unverändert zum heutigen Zustand). Wie das Prompt kurze/uninformative Captions behandelt, entscheidet der Impl-Plan — dieses Design legt keinen Filter fest.
- Keine Heuristik zur Caption-Propagation zwischen Geschwister-Elementen — siehe Nicht-Scope.

## Nicht-Scope (explizit ausgeschlossen)

Projekt-Prinzip: **keine Tool-spezifischen Heuristiken, bevor Benchmarks entscheiden, welches Tool bleibt.**

- **MinerU-Heading-Misklassifikation** (z. B. `2.3.2. Material Modeling …` landet als `text` mit `conf=0.57` statt `heading`). Hängt an MinerU's `title` vs. `text`-Entscheidung, nicht in unserem Scope.
- **Deko-Icon-Filter** (z. B. `check for updates`-Logo `5ad2801f7ebeb1c8` auf Seite 0, 54×20 pt, `conf=0.53`). Ein Größen-/Confidence-Gate wäre trivial, aber die Schwellwerte sind tool-abhängig.
- **Sub-Label-concat mit Haupt-Caption** (z. B. Seite 2, Region 27: `caption = "(b)\nFigure 3. Stress–strain curves …"`). Hängt an MinerU's `chart_caption`-Block-Zusammenfassung in `_collect_block_text` (`extraction/adapters/mineru25_segmenter.py:357-365`).
- **Caption-Propagation für Sub-Figures** (z. B. Seite 6, Element `b83b341b9ca31016` mit Caption nur `"(a)"` — die Haupt-Caption `"Figure 8. Oil pan after cold stamping …"` sitzt auf einem Geschwister-Element). Folge: die zwei Sub-Figures im manuellen Prüf-Satz werden durch Plan #4 nicht gelöst; das ist als bekannte Limitation markiert.
- **End-to-End-Validierung** Sidecars ↔ `content_list.json` als Pipeline-Selbstcheck.
- **Phase-2-Artefakte** (`section_path`, `mentions`, `relations`, `document_rich.json`).

## Test-Plan

### Neue Unit-Tests

- `tests/test_stages_extract_text.py`:
  - role-match: Stage 1 hat Sidecar geschrieben → Stage 2 überspringt, Sidecar wird nicht überschrieben.
  - role-mismatch für `text`: Crop (kleiner als Vollseite) wird an Extractor gereicht, Sidecar entsteht.
  - role-mismatch für `table`: Table-Sidecar entsteht in Stage 2 statt still zu verschwinden.
  - role-mismatch für `formula`: Formula-Sidecar entsteht in Stage 2 statt still zu verschwinden.
  - Caption aus der Region bleibt in allen Pfaden im Sidecar erhalten.
- `tests/test_cli.py`:
  - `cfg.output_dir` greift, wenn `--out` fehlt.
  - `--out` überschreibt `cfg.output_dir`.
- `tests/test_stages_describe_figures.py`:
  - `describe()` wird mit `caption=<string>` aufgerufen, wenn die Region Caption trägt.
  - `describe()` wird mit `caption=None` aufgerufen, wenn keine Caption da ist.

### Manueller Prüf-Satz (Variante 2)

Vier IDs aus `jmmp-09-00199-v2`, mit klarer Kennzeichnung der Limitation:

| element_id         | Seite | Caption im Sidecar                            | Erwartung dieses Passes                                                                        |
|--------------------|-------|-----------------------------------------------|------------------------------------------------------------------------------------------------|
| `eaa22bc5a19886da` | 2     | `"Figure 2. Tensile test specimens …"`         | Beschreibung konsistent zur Caption (keine „cutting tools"-Halluzination).                     |
| `a350c20659b04f17` | 8     | `"Figure 11. Initial blank size …"`            | Beschreibung konsistent zur Caption.                                                           |
| `b83b341b9ca31016` | 6     | `"(a)"` (Sub-Figure)                           | **Bekannte Limitation**: Prompt hat keinen echten Kontext. Post-Benchmark mit Caption-Propagation adressiert. |
| `f7c30b7a776b2189` | 6     | `"(b)"` (Sub-Figure)                           | Wie oben.                                                                                      |

Akzeptanz für die ersten beiden: Beschreibung darf allgemein sein, aber **nicht klar widersprüchlich** zu Caption oder Crop. Die letzten beiden bleiben als Beleg im Test-Set, damit der nächste Pass den Fortschritt messen kann.

### Quality Gates

```bash
venv/bin/pytest -q
venv/bin/ruff check extraction
venv/bin/mypy extraction
```

`mypy`-Striktheit wird **nicht** gelockert; betroffene Test-/Stub-Funktionen werden bei Bedarf annotiert.

## Assumptions / Defaults

- Korrektheit vor Qualitäts-Tuning.
- Öffentliche Output-Schemas bleiben stabil.
- 4-stufiger User-Workflow bleibt; keine zusätzliche Table/Formula-Stage.
- Keine aggressiven Filter für Deko-Elemente, keine Caption-Zuordnungs-Heuristik in diesem Pass.
- Keine Breaking Changes an Dateinamen oder Output-Pfaden innerhalb eines bestehenden Runs.

## Verifikation gegen aktuellen Code (2026-04-24)

- Stage-2-Scope: `extraction/stages/extract_text.py:17`
- `TextExtractor` Signatur/Docstring: `extraction/interfaces.py:40`, `extraction/stages/extract_text.py:100-101`, `extraction/adapters/olmocr2_text.py:104-107`
- `FigureDescriptor`: `extraction/interfaces.py:67`, `extraction/adapters/qwen25vl_figure.py:93-117`
- `--out` default: `extraction/__main__.py:33`
- Passthrough-Extractoren: `extraction/adapters/mineru25_segmenter.py:109-173`
- Caption-Muster in Stage 2: `extraction/stages/extract_text.py:102-103`
- `segmentation.json`-Struktur: `extraction/stages/segment.py:123-129`
- Doku-Drift: `.claude/CLAUDE.md:35`, `docs/extraction_output.md:3,55,146-151`, `docs/architecture.md:189,456`

Der im Vorgänger-Plan erwähnte „veraltete GPU-Integrationstest, der `python -m extraction extract …` aufruft" existiert **nicht** in `extraction/tests/` (grep-geprüft). Dieser Punkt wurde aus dem Test-Plan entfernt.
