# Extraction Hardening Design

## Ziel

Den Extraction-Block in einen Zustand bringen, in dem er:

- technisch korrekt arbeitet (Bounding-Boxes, Crops, IDs)
- die Config wirklich respektiert, statt im Code eigene Tool-Heuristiken zu verstecken
- sauber paketiert ist (`pip install -e .` funktioniert, Dependencies stimmen)
- klare Verträge zwischen Segmenter, Extractor, Pipeline und Output hat
- fail-safe gegen versehentliches Überschreiben vorhandener Output-Ordner ist

**Nicht im Scope:** `embedding/` und `indexing/` bleiben Platzhalter.
`document_rich.json`, Section-Paths, Mentions und Multi-Page-Merge bleiben Phase 2.

## Auslöser

Der aktuelle Stand hat eine Reihe Bugs und unsauberer Verträge, die vor der
nächsten Ausbaustufe gefixt werden müssen:

- Bounding-Boxes werden von Segmentern in PDF-Points geliefert, aber auf
  Pixel-Bildern gecroppt → falsche Crops, falsche Extractor-Inputs,
  falsche gespeicherte `.png`-Dateien.
- `element_id` basiert auf dem Dateipfad → derselbe PDF an verschiedenen
  Pfaden liefert verschiedene IDs.
- Top-Level-`dpi` in `ExtractionConfig` wird nirgendwo weitergereicht.
- Die Merge-Regel der Pipeline nutzt Segmenter-Content pauschal. Damit ist
  die Config-Austauschbarkeit für Extractor-Rollen effektiv stumm.
- Der `figure_descriptor` läuft nie, wenn der Segmenter eine Caption liefert,
  weil der Content-Check dann einfach durchreicht.
- Kein Schutz gegen Re-Runs in denselben Output-Ordner; alte Sidecars
  vermischen sich mit neuen.
- Echte Confidence-Werte von MinerU werden verworfen und hart auf `1.0`
  gesetzt → `confidence_threshold` ist ein Placebo.
- `pyproject.toml` hat keine Package-Discovery-Config und listet `pymupdf`
  nicht als Dependency, obwohl der Default-Renderer es braucht.
- Noop-Text-Extractor erzeugt leere Text-Elemente, die als echte Elemente
  geschrieben werden.

## Design-Entscheidungen

### 1. Koordinatensystem

**Vertrag:** Alle `bbox`-Werte in `Region` und `Element` sind in
**PDF-Points**, Origin top-left, DPI-unabhängig.

- Beide aktuellen Segmenter (MinerU 2.5 via `middle_json`, PyMuPDF via
  `get_text("dict")`) liefern bereits in Points.
- Die Pipeline skaliert bbox **nur beim Cropping**:
  `scale = dpi / 72`. Der Skalierungsschritt passiert an genau einer
  Stelle (`OutputWriter.crop_region`).
- Crops werden auf Bildgrenzen geklemmt (negative Werte auf 0,
  Überläufe auf Bildbreite/-höhe). Damit erzeugen leicht überschießende
  Bboxes keinen kaputten Crop.

**Begründung:** Points sind DPI-unabhängig und bleiben reproduzierbar,
wenn die Pipeline später mit anderer DPI re-rendert. Segmenter bleiben
DPI-agnostisch (sie kennen nur die PDF, nicht den Render-Kontext).

**Dokumentation:** neuer Abschnitt „Koordinatensystem" in
`docs/extraction_output.md`; Docstrings an `Region.bbox`, `Element.bbox`,
`OutputWriter.crop_region`.

### 2. Element-ID

**Vertrag:**
```
element_id = sha256(
    doc_id + ":" + page + ":" + region_type +
    ":" + round(x0) + "," + round(y0) + "," + round(x1) + "," + round(y1)
)[:16]
```

- Pfadunabhängig: derselbe PDF-Inhalt liefert überall dieselbe ID.
- Bbox-basiert statt sequenzbasiert: stabil gegen ML-Nondeterminismus in
  der Region-Reihenfolge (MinerU garantiert laut eigener Doku keine
  stabile Reihenfolge bei komplexen Layouts).
- Rundung auf ganze Points reicht (Float-Rauschen unterhalb eines Points
  ist praktisch irrelevant).

**Kollisionsrisiko:** Nur bei zwei exakt identischen bboxes mit
demselben Typ auf derselben Seite — extrem unwahrscheinlich.

### 3. `source_file`

**Vertrag:** `source_file = pdf_path.name` (nur Dateiname, ohne Pfad).

Damit ist `content_list.json` zwischen Rechnern portabel.

### 4. Merge-Regel — Config diktiert

**Grundprinzip:** Die Config weist jedem Role genau ein Tool zu. Die
Pipeline **gehorcht der Config**. Sie entscheidet nicht selbst, welches
Tool „besser" ist.

**Datenquellen pro Feld:**

| Feld                | Quelle                                |
|---------------------|---------------------------------------|
| `bbox`, `page`, `type`, `reading_order` | Segmenter               |
| `caption`           | Segmenter (Layout-Daten)              |
| `image_path`        | Pipeline (aus dem Crop)               |
| `text`              | `text_extractor`                      |
| `markdown`          | `table_extractor`                     |
| `latex`             | `formula_extractor`                   |
| `description`       | `figure_descriptor`                   |

**Ablauf pro Region in `Pipeline._extract_region`:**

```
1. Pipeline bestimmt das Role-Tool für den Region-Typ (laut Config).
2. Wenn role_tool.tool_name == segmenter.tool_name:
       Pipeline übernimmt region.content.<relevantes_feld> (Optimierung:
       kein Re-Run, weil semantisch identisch).
   Sonst:
       Pipeline cropped das gerenderte Seitenbild, übergibt Crop an das
       Role-Tool, übernimmt dessen Output.
3. Layout-Felder (caption) bleiben immer aus dem Segmenter — unabhängig
   davon, welches Tool den Content liefert.
4. Extraktions-Output mit leerem Pflichtfeld (z. B. leerer Text) wird
   als „keine Daten" gewertet — siehe Punkt 7.
```

**Beispiele:**

| Config                                           | Verhalten                                                                                 |
|--------------------------------------------------|-------------------------------------------------------------------------------------------|
| `segmenter: mineru25`, `table_extractor: mineru25` | Pipeline nutzt MinerUs internes Markdown (Tool-Match-Optimierung).                       |
| `segmenter: mineru25`, `table_extractor: anderes` | Pipeline cropped, lässt `anderes` laufen, übernimmt dessen Output. MinerUs Markdown wird verworfen. |
| `segmenter: mineru25`, `table_extractor: noop`   | Noop-Output wird übernommen (leer). Tabellen-Elemente haben `content.markdown = None`. Kein Fallback. |
| `segmenter: pymupdf_text`, `text_extractor: olmocr2` | PyMuPDF-Text-Content aus der Region wird verworfen, OlmOCR läuft.                     |

**Figure-Sonderfall:** MinerU liefert für Figures/Charts/Drawings oft nur
eine Caption (Layout), keine Description. Unter der obigen Regel:
- Caption → immer aus Segmenter, wenn vorhanden
- Description → `figure_descriptor`-Tool. Wenn `figure_descriptor: noop`,
  bleibt description leer — kein automatischer Fallback, kein
  Re-Run-Heuristik.

**`extractor`-Feld pro Element:** Name des Role-Tools, das den primären
Content für diesen Element-Typ geliefert hat. Einzelner String, kein
zusammengesetztes Label. Der Segmenter-Name steht in
`content_list.json:segmentation_tool` auf Dokument-Ebene.

**Status:** Diese Regel ist **v1-Default**. Welche Tool-Kombination pro
Role optimal ist, wird durch einen Benchmark entschieden (siehe
Backlog-Eintrag), nicht durch Pipeline-Logik.

### 5. Output-Isolation (Fail-Safe)

**Vertrag:** `ExtractionPipeline.run()` bricht **sofort** ab, wenn der
Output-Ordner bereits Extraction-Artefakte enthält.

**Geprüfte Artefakte:**
- `content_list.json`
- `segmentation.json`
- `pages/` (Verzeichnis und nicht-leer)

**Fehlermeldung** nennt die zwei Handlungsoptionen: anderen
Output-Ordner wählen, oder Artefakte manuell entfernen. **Kein
automatisches Löschen, kein `--overwrite`-Flag** — Sicherheit vor
Komfort.

`rebuild`-Sub-Command bleibt unverändert, weil er per Design auf
vorhandene Sidecars angewiesen ist.

**Wo:** Der Check sitzt in `Pipeline.run()`, nicht in `__main__.py`.
Damit greift er auch bei programmatischer Nutzung (Tests,
Benchmark-Runner, Skripte).

**`document_rich.json` bewusst nicht im Check:** Wird in Phase 1 nicht
produziert. Doku hält fest: sobald Phase 2 die Datei einführt, wird sie
zum Fail-Safe-Check hinzugefügt — mit genau der gleichen Semantik.

### 6. Confidence als aktives Filter

**Vertrag:** Jeder Segmenter-Adapter liefert echte Confidence-Werte pro
Region, statt hart `1.0` zu setzen.

- `MinerU25Segmenter` liest `middle_json.layout_dets[*].score` aus und
  mappt die Werte per bbox-Matching auf die `para_blocks`.
- `PyMuPDFTextSegmenter` setzt weiterhin `1.0` — PyMuPDF ist
  regelbasiert und hat keine ML-Confidence. Wird im Code kommentiert.
- `confidence_threshold` in `ExtractionConfig` (Default `0.3`) ist
  dadurch wirksam. Tunable via Config.

**Dokumentation:** neue Datei `docs/writing_adapters.md` mit einer
Checkliste für neue Adapter. Punkt 1: echte Confidence auslesen, nicht
hart setzen. Damit ist bei jedem Tool-Wechsel klar, dass der
Confidence-Filter mitgedacht werden muss.

### 7. Noop- und Leer-Content-Verhalten

- Text- oder Heading-Elemente mit leerem oder whitespace-only
  `content.text` werden **nicht** als Element in die Sidecars /
  `content_list.json` geschrieben.
- Tabellen-Elemente ohne `markdown` und ohne `text` werden gedroppt.
- Formel-Elemente ohne `latex` und ohne `text` werden gedroppt.
- Visuelle Elemente (`figure`, `diagram`, `technical_drawing`) dürfen
  persistieren, solange ein `image_path` vorhanden ist — der Crop selbst
  ist ein verwertbares Artefakt, auch ohne Description.

### 8. Packaging

`pyproject.toml`:

- `setuptools`-Package-Discovery explizit auf `extraction*` beschränkt.
- `outputs*`, `_archive*`, `embedding*`, `indexing*`, `tests*` explizit
  ausgeschlossen.
- `pymupdf` als Runtime-Dependency aufgenommen (Default-Renderer braucht
  es).
- Optionales Extra `[gpu]` für MinerU / Transformers / Torch /
  BeautifulSoup. CUDA-spezifische Torch-Installation wird in der README
  dokumentiert statt im Code erraten.

### 9. Config und DPI

- Top-Level `dpi` wird an den Renderer durchgereicht, wenn
  `adapters.<renderer>.dpi` nicht explizit gesetzt ist.
- Adapter-spezifische Config hat immer Vorrang vor Top-Level-Defaults.
- Production-Defaults bleiben: `mineru25`, `olmocr2`, `qwen25vl`
  (GPU vorhanden).
- Eine CPU-Beispielconfig wird in der README dokumentiert, aber nicht
  als Code-Default gesetzt.

### 10. Cleanups

- `reading_order_index` wird nur noch einmal gesetzt (in
  `OutputWriter.build_content_list`), nicht redundant vorab in der
  Pipeline.
- `_extract_region` instanziert `OutputWriter` einmal pro Pipeline-Lauf,
  nicht einmal pro Region.

## Public Behavior

- `python -m extraction extract ...` bricht bei vorhandenen Artefakten
  im Output-Ordner ab. Neuer Output-Ordner oder manuelles Aufräumen
  vorher notwendig.
- `python -m extraction rebuild ...` unverändert.
- `content_list.json` Schema-Version bleibt `1.0`. Keine neuen
  Pflichtfelder.
- `element_id` ändert sich **einmalig** gegenüber dem alten
  pfadbasierten Verfahren. Akzeptabel, weil kein Downstream-Block
  produktiv auf den alten IDs aufbaut.
- `source_file` ändert sich von vollem Pfad auf Dateiname.
- `confidence` bei MinerU-Elementen enthält ab jetzt echte Werte
  (vormals hart `1.0`).
- Crops in `outputs/<doc>/pages/<n>/*.png` werden ab jetzt korrekt
  positioniert (vorher falsch durch Points-statt-Pixel).

## Test Plan

### Packaging
- `pip install -e .` gelingt; Package-Discovery kollidiert nicht mit
  `outputs/`, `_archive/`, `embedding/`, `indexing/`.

### Config & DPI
- `ExtractionConfig(dpi=300)` führt zu Renderer-DPI `300`.
- `adapters.pymupdf.dpi: 450` überschreibt Top-Level `dpi: 300`.

### Fail-Safe
- Pipeline bricht bei vorhandenem `content_list.json`,
  `segmentation.json` oder nicht-leerem `pages/` ab, mit klarer
  Fehlermeldung.
- `rebuild` funktioniert weiterhin mit bestehendem `content_list.json`
  und Sidecars.

### Merge-Regel
- `segmenter: mineru25`, `table_extractor: mineru25`: MinerUs internes
  Markdown im Element, **kein** Aufruf des Table-Extractors über den
  Crop (per Mock verifiziert, dass `extract` nicht gecallt wird).
- `segmenter: mineru25`, `table_extractor: <mock_tool>`: MinerUs
  internes Markdown wird verworfen, `<mock_tool>` gecallt, dessen
  Output im Element.
- `segmenter: mineru25`, `table_extractor: noop`: Element hat leeres
  `markdown` und wird nach Regel 7 gedroppt.
- Figure mit MinerU-Caption + `figure_descriptor: <mock_desc>`:
  Element hat sowohl `caption` (aus Segmenter) als auch `description`
  (aus Mock-Descriptor).
- Figure mit MinerU-Caption + `figure_descriptor: noop`: Element hat
  `caption`, aber keine `description`.

### Leere Contents
- Text-Element mit leerem Text (`""` oder `"   "`) wird gedroppt.
- Table-Element ohne markdown/text wird gedroppt.
- Figure-Element mit nur `image_path` wird behalten.

### IDs
- Identische PDF-Datei an zwei verschiedenen Pfaden erzeugt
  identische `element_id`s (bbox-basiert).
- Zwei Regionen mit verschiedenem Typ an identischer bbox haben
  unterschiedliche `element_id`s.

### Crops
- Crop mit bbox außerhalb des Bildes wird auf Bildgrenzen geklemmt
  (keine negativen oder überlaufenden Koordinaten).
- Crop bei DPI=150 nutzt Skalierung `150/72` korrekt: ein bbox
  `[100, 200, 200, 300]` in Points wird auf
  `[~208, ~417, ~417, ~625]` in Pixeln gecroppt.

### Confidence
- MinerU-Adapter-Test: Regionen tragen Werte aus `layout_dets.score`,
  nicht hart `1.0`.

### Regression
- `venv/bin/python -m pytest -q`
- `venv/bin/ruff check extraction`
- `venv/bin/mypy`

### Smoke
- Leichter PyMuPDF-CPU-Run auf einem vorhandenen Test-PDF erzeugt
  `content_list.json`, `segmentation.json`, `pages/*/page.png` und
  Sidecars.
- Integration: markierter MinerU-/GPU-Test bleibt separat und ist
  nicht Teil des normalen `pytest -q`.

## Dokumentation

### `docs/extraction_output.md`
- Neuer Abschnitt „Koordinatensystem": bbox in Points,
  Pixel-Umrechnung beim Cropping.
- Abschnitt „Merge-Regeln" überarbeitet: Config diktiert, keine
  Tool-Heuristik, Tool-Match-Optimierung dokumentiert.
- Abschnitt „IDs" aktualisiert: bbox-basiert, pfadunabhängig.
- Hinweis: `table_extractor: mineru25` in der Default-Config nutzt
  MinerUs internen Content via Tool-Match — das ist Optimierung, kein
  Design-Constraint. Tool-Vergleich pro Role siehe Benchmark-Backlog.
- Hinweis: Phase 2 wird `document_rich.json` einführen; sobald das
  geschieht, wird die Datei zum Fail-Safe-Check hinzugefügt.

### `README.md` (neu/aktualisiert)
- Installation mit `pip install -e .` und `pip install -e .[gpu]`
- CUDA-Torch-Hinweis
- CPU-Beispielconfig (ohne GPU)
- CLI-Nutzung: `extract` und `rebuild`
- Quality Gates: `pytest -q`, `ruff check extraction`, `mypy`

### `docs/writing_adapters.md` (neu)
Checkliste für neue Adapter. Pflichtpunkte:
1. Echte Confidence mappen (nicht hart `1.0`).
2. bbox in PDF-Points liefern.
3. Tool-Name als Property konsistent setzen.
4. Beim Wechsel der Role-Zuordnung ist die Merge-Regel zu beachten
   (Layout vs. Content Felder).

### Alt-Dokumente markieren
- `tasks/extraction_completion_plan.md` und `tasks/todo.md` verweisen
  auf die alte `src/…`-Struktur. Als Altstand/Archiv markieren, nicht
  mehr pflegen.

### `.claude/CLAUDE.md`
Nur aktualisieren, wenn eine lokale Arbeitsanweisung mit dem neuen
Stand kollidiert. Sonst unverändert lassen.

## Backlog-Einträge (English, per user convention)

**Bbox-overlap deduplication** (nice-to-have, defensive):
Detect overlapping regions from a single segmenter (IoU > 0.8 as
start threshold), keep the one with higher confidence, discard the
rest. Trigger for implementation: once duplicates are observed in
real PDF runs. Parked until evidence arrives.

**Per-role extractor benchmark** (extends existing entry
„Compare PDF Extraction Tools" from 2026-04-16):
Run the pipeline with different `table_extractor`,
`formula_extractor`, `figure_descriptor` combinations on a fixed
test corpus. Compare outputs against a hand-curated ground truth.
Output: a recommendation table per role. Unblocks data-driven
default changes in `ExtractionConfig`.

## Nicht im Scope (Phase 2 oder später)

- `embedding/` und `indexing/` bleiben Platzhalter, kein Code.
- `document_rich.json` mit `sections` und `relations`.
- `section_path` pro Element.
- `mentions` pro Text-Element.
- `caption_ref` als Pointer.
- Multi-Page-Element-Zusammenführung.
- Backwards-Compatibility-Layer für alte pfadbasierte `element_id`s.
- `--overwrite`-Flag oder automatisches Löschen.
- Automatische Duplikat-Erkennung (geparkt im Backlog).

## Annahmen

- GPU-Production-Pfad ist erwünscht; Code-Defaults bleiben MinerU /
  OlmOCR / Qwen.
- Kein Downstream-Block (Embedding, Indexing) baut produktiv auf den
  alten pfadbasierten `element_id`s auf — ID-Wechsel ist also sicher.
- PyMuPDF als Default-Renderer akzeptiert; ein alternativer Renderer
  wäre in dieser Iteration unnötig.
