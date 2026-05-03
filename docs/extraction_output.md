# Extraction Output Format

Stabile Spezifikation des Outputs, den der 4-stufige Workflow (`python -m extraction segment|extract-text|describe-figures|assemble`) produziert.
Dieser Kontrakt ist die Schnittstelle zwischen dem Extraction-Block (Layer 1) und allen
nachgelagerten Blöcken (Embedding, Storage, Agent). Tools können ausgetauscht werden, das
Format bleibt stabil.

Code-Quelle dieser Definitionen: `extraction/models.py` (Pydantic-Modelle). Wenn Code und
Spec voneinander abweichen: **der Code gewinnt**, die Spec wird aktualisiert.

## Ordnerstruktur

Beispiel für eine 3-seitige PDF mit Text, einer Tabelle und einer Abbildung:

```
outputs/<run>/
├── content_list.json                 ← gemergte, flache Elementliste nach assemble
├── segmentation.json                 ← rohe Regionen vor Extraktion (Debug)
├── .stages/
│   ├── segment.done
│   ├── extract-text.done
│   ├── describe-figures.done
│   └── assemble.done
└── pages/
    ├── 0/
    │   ├── page.png                  ← gerenderte Seite (DPI aus Config, default 150)
    │   ├── <el_id>_heading.json      ← Text-/Heading-Element nur als JSON
    │   └── <el_id>_text.json
    ├── 1/
    │   ├── page.png
    │   ├── <el_id>_text.json
    │   ├── <el_id>_table.json        ← visuelles Element zusätzlich mit Crop
    │   └── <el_id>_table.png
    └── 2/
        ├── page.png
        ├── <el_id>_figure.json
        ├── <el_id>_figure.png
        └── <el_id>_text.json
```

Regeln:

- Jedes Element bekommt einen JSON-Sidecar im Ordner seiner Seite.
- Visuelle Typen (`table`, `formula`, `figure`, `diagram`, `technical_drawing`)
  bekommen zusätzlich ein PNG-Crop mit demselben `<el_id>_<type>`-Stamm.
- Nicht-visuelle Typen (`text`, `heading`) bekommen **nur** die JSON.
- `pages/<N>/page.png` ist die komplette gerenderte Seite (nützlich zum visuellen
  Vergleich und als Basis für spätere Region-Crops).
- `.stages/<stage>.done` und `.stages/<stage>.error` halten den Stand der
  vier Stage-Kommandos fest.

## Quelle der Wahrheit

Die **Per-Element-JSONs im Page-Ordner** sind die Quelle der Wahrheit.

`content_list.json` ist eine **deterministische, gemergte Sicht** dieser Sidecars —
sortiert nach `(page, reading_order_index, element_id)` und mit neu durchnummeriertem
`reading_order_index` über alle Seiten hinweg. Die Datei lässt sich jederzeit aus den
Sidecars rekonstruieren:

```bash
python -m extraction assemble outputs/<run>/
```

Der Vorteil: Wenn ein einzelner Extractor verbessert werden soll (z. B. Formel-LaTeX
genauer), können gezielt einzelne Sidecars neu erzeugt oder editiert und danach
`content_list.json` in einem Schritt neu gebaut werden — ohne die komplette
Extraktionspipeline erneut laufen zu lassen.

## Element-Schema

Jeder Sidecar und jeder Eintrag in `content_list.json:elements` folgt diesem Schema
(Pydantic-Modell: `extraction.models.Element`):

```jsonc
{
  "element_id": "sha256-16-hex",
  "type": "text | heading | table | formula | figure | diagram | technical_drawing",
  "page": 1,
  "bbox": [x0, y0, x1, y1],
  "reading_order_index": 5,
  "section_path": [],
  "confidence": 0.91,
  "extractor": "mineru25",
  "content": {
    "text": "...",
    "markdown": "...",
    "html": "<table>...</table>",
    "latex": "...",
    "image_path": "pages/1/<el_id>_table.png",
    "description": "...",
    "caption": "...",
    "caption_position": "above | below",
    "footnotes": [{"text": "..."}],
    "markers": [{"value": "4.2", "marker": "a"}]
  }
}
```

Pflichtfelder: `element_id`, `type`, `page`, `bbox`, `reading_order_index`,
`confidence`, `extractor`, `content`.

Optionale Felder in `content` — welches gesetzt ist, hängt vom `type` ab:

| type                | content-Felder die gesetzt sein sollten                  |
|---------------------|----------------------------------------------------------|
| text                | `text`                                                   |
| heading             | `text`                                                   |
| table               | `image_path`, idealerweise `markdown`, `html`, `text`, ggf. `caption`, ggf. `caption_position`, ggf. `footnotes`, ggf. `markers` |
| formula             | `image_path`, idealerweise `latex`, `text`               |
| figure              | `description`, `image_path`, ggf. `caption`              |
| diagram             | `description`, `image_path`, ggf. `caption`              |
| technical_drawing   | `description`, `image_path`, ggf. `caption`              |

`markdown` ist eine flache Pipe-Darstellung für Text-basierte Konsumenten
(Embedding, Suche). `html` behält die Original-Struktur des Segmenters
inklusive `rowspan`/`colspan` für Konsumenten, die die Tabellen-Hierarchie
brauchen (Rendering, strukturierte Queries). `text` ist bei Tabellen
identisch zu `markdown`.

Felder die nicht gesetzt sind, werden beim Schreiben **weggelassen** (`exclude_none=True`),
nicht als `null` serialisiert.

`section_path` ist im Schema enthalten, wird aber in Phase 1 (diese Iteration) noch
nicht befüllt — bleibt `[]`. Siehe Phase 2 weiter unten.

## content_list.json

Gemergte Sicht, direkter Input für den Embedding-Block.

```jsonc
{
  "doc_id": "sha256-16-hex",                // Hash des PDF-Inhalts
  "source_file": "druckbericht.pdf",
  "total_pages": 3,
  "schema_version": "1.0",
  "segmentation_tool": "mineru25",          // welcher Segmenter lief
  "pages": [
    {"page": 0, "image_path": "pages/0/page.png", "element_ids": ["el001", "el002"]},
    {"page": 1, "image_path": "pages/1/page.png", "element_ids": ["el003", "el004", "el005"]},
    {"page": 2, "image_path": "pages/2/page.png", "element_ids": ["el006", "el007"]}
  ],
  "elements": [ /* Element-Schema wie oben, sortiert nach reading_order_index */ ]
}
```

`schema_version` lebt auf Ebene der gemergten Datei (nicht pro Element), weil das Schema
für den ganzen Run einheitlich ist.

## segmentation.json

Rohe Regionen des Segmenters **vor** jeder Content-Extraktion. Debug- und
Vergleichszweck (zwei Segmenter gegeneinander laufen lassen, Layout-Treffer prüfen).

```jsonc
{
  "doc_id": "sha256-16-hex",
  "source_file": "druckbericht.pdf",
  "total_pages": 3,
  "segmentation_tool": "mineru25",
  "render_dpi": 150,
  "stage_config": {
    "renderer": "pymupdf",
    "segmenter": "mineru25",
    "text_extractor": "mineru25",
    "table_extractor": "mineru25",
    "formula_extractor": "mineru25",
    "figure_descriptor": "qwen25vl",
    "confidence_threshold": 0.3,
    "renderer_config": {},
    "segmenter_config": {}
  },
  "regions": [
    {"page": 0, "bbox": [x0, y0, x1, y1], "region_type": "heading", "confidence": 0.99, "reading_order_index": 0},
    {"page": 0, "bbox": [x0, y0, x1, y1], "region_type": "text",    "confidence": 0.95, "reading_order_index": 1},
    {"page": 1, "bbox": [x0, y0, x1, y1], "region_type": "table",   "confidence": 0.93, "reading_order_index": 2, "content": {"markdown": "..."}}
  ]
}
```

Wenn der Segmenter bereits Content mitliefert (MinerU füllt z. B. Text,
Tabellen-HTML/-Markdown, Formel-LaTeX und Captions), steht dieser hier mit drin.
`segment` schreibt daraus direkt Element-Sidecars, wenn der konfigurierte
Role-Extractor denselben `tool_name` wie der Segmenter hat.

`render_dpi` ist die effektive DPI des gerenderten `page.png`. Nachgelagerte
Stages verwenden diesen Wert beim Cropping, nicht die eventuell inzwischen
geänderte aktuelle Config. `stage_config` ist ein kleiner Fingerprint der
für Stage 1 relevanten Adapter- und Threshold-Auswahl; `segment` nutzt ihn,
um stale Outputs mit existierendem `segment.done` sicher zu erkennen.

## Merge-Regeln

Die Pipeline entscheidet pro Region **nicht selbst**, welches Tool den Content
liefert. Die Config weist jedem Role (text, table, formula, figure) genau ein
Tool zu. Die Stages gehorchen.

Datenquellen pro Feld:

| Feld                                       | Quelle                  |
|--------------------------------------------|-------------------------|
| `bbox`, `page`, `type`, `reading_order`    | Segmenter               |
| `caption`                                  | Segmenter (Layout)      |
| `caption_position` (nur `table`)           | Segmenter (Layout)      |
| `footnotes` (nur `table`)                  | Segmenter (Layout)      |
| `markers` (nur `table`)                    | Segmenter (Layout, aus `<sup>` im HTML) |
| `image_path`                               | Pipeline (aus dem Crop) |
| `text`                                     | `text_extractor`        |
| `markdown`, `html`                         | `table_extractor`       |
| `latex`                                    | `formula_extractor`     |
| `description`                              | `figure_descriptor`     |

Ablauf pro Region im staged Workflow:

1. `segment` rendert Seiten, schreibt `segmentation.json` und prüft pro Region,
   ob das konfigurierte Role-Tool denselben Namen wie der Segmenter hat.
2. Bei Tool-Match und vorhandenem `region.content` schreibt `segment` direkt
   den Element-Sidecar. Für visuelle Typen wird zusätzlich ein Crop gespeichert.
3. `extract-text` behandelt Text, Heading, Tabelle und Formel nur dort, wo noch
   kein Sidecar existiert oder `--force` gesetzt ist. Dazu cropped die Stage das
   gespeicherte `page.png` und ruft das konfigurierte Role-Tool auf.
4. `describe-figures` behandelt Figure, Diagram und Technical Drawing analog
   mit dem konfigurierten `figure_descriptor`.
5. `caption` aus dem Segmenter bleibt erhalten und wird auf den Extractor-Output
   gelegt. `caption_position`, `footnotes` und `markers` (alle drei nur `table`,
   abgeleitet aus den `table_caption`/`table_footnote`-Sub-Blöcken des Segmenters
   bzw. aus `<sup>`-Tags im Tabellen-HTML) bleiben nur unter Tool-Match
   (Passthrough) erhalten — bei abweichendem `table_extractor` propagiert die
   Stage sie heute nicht weiter (siehe Backlog).
6. Ein Role-Tool-Output mit leerem Pflichtfeld wird gedroppt, wenn keine
   visuelle Evidenz existiert. Tabellen und Formeln dürfen als Crop-only
   Fallback persistieren, solange ein `image_path` vorhanden ist; Text und
   Headings ohne Text werden gedroppt.

Beispiele:

| Config                                           | Verhalten                                                                                 |
|--------------------------------------------------|-------------------------------------------------------------------------------------------|
| `segmenter: mineru25`, `table_extractor: mineru25` | Tool-Match — MinerUs internes Markdown wird direkt übernommen.                          |
| `segmenter: mineru25`, `table_extractor: anderes` | Crop wird durch `anderes` geschickt, MinerUs Markdown verworfen. Caption bleibt erhalten. |
| `segmenter: mineru25`, `figure_descriptor: noop` | Figure-Element hat `caption` aus dem Segmenter, aber keine `description`.               |

Welche Tool-Kombination pro Role optimal ist, wird durch Benchmarks
entschieden, nicht durch Pipeline-Heuristik. Wenn Segmenter und Role-Tool
denselben Registry-Namen haben, nutzt die Pipeline die
Tool-Match-Optimierung — das ist Effizienz, kein Design-Constraint.

## Reading Order

- Pro Seite legt der Segmenter die Reihenfolge fest (top-to-bottom, left-to-right —
  für gängige Layouts funktioniert das; MinerU berücksichtigt zusätzlich
  Mehrspaltigkeit).
- Über Seiten hinweg: Seite 0 → Seite 1 → …
- `reading_order_index` auf jedem Element ist eine **globale**, durchgehende
  Nummerierung ab 0, stabil nach Confidence-Filterung.
- Tie-Breaker bei gleicher `(page, reading_order_index)` in der Merge-Sicht:
  lexikografische Sortierung der `element_id`.

## Koordinatensystem

`bbox` in jedem `Region` und jedem `Element` ist in **PDF-Points**, Origin
oben-links, DPI-unabhängig. Beide mitgelieferten Segmenter (MinerU via
`middle_json`, PyMuPDF via `get_text("dict")`) liefern bereits in Points.

Die Pipeline skaliert bbox nur beim Cropping auf das gerenderte Seitenbild:
`scale = dpi / 72`. Passiert an genau einer Stelle (`OutputWriter.crop_region`).
Negative Werte und Überläufe werden auf Bildgrenzen geklemmt, damit ein leicht
überschießendes bbox keinen kaputten Crop erzeugt.

Vorteil: Segmenter bleibt DPI-agnostisch. Wenn später mit anderer DPI
re-rendert wird, braucht weder Segmenter noch Output-Format geändert werden —
nur der eine Skalierungsschritt.

## IDs

- `doc_id`: SHA-256 des PDF-Inhalts, auf 16 Hex-Zeichen gekürzt.
- `element_id`: SHA-256 über
  `"<doc_id>:<page>:<region_type>:<round(x0)>,<round(y0)>,<round(x1)>,<round(y1)>"`,
  auf 16 Hex-Zeichen gekürzt. Pfadunabhängig: derselbe PDF-Inhalt liefert
  überall dieselben IDs. Bbox-basiert statt sequenzbasiert — stabil, auch
  wenn ein ML-Segmenter die Regionen nicht exakt in derselben Reihenfolge
  liefert.

## Was Phase 2 bringt (nicht in dieser Iteration)

Diese Felder und Artefakte sind bewusst aus Phase 1 ausgenommen und folgen später:

- `document_rich.json` mit
  - `sections`: hierarchischer Kapitelbaum aus den Heading-Elementen (oder PDF-Outline)
  - `relations`: `captioned_by` (Figur ↔ Caption) und `refers_to` (Text ↔ Tabelle, etc.)
- `section_path` pro Element — fällt automatisch ab, sobald der Kapitelbaum existiert
- `mentions` pro Text-Element — Parser sucht nach „Tabelle 1", „Abb. 3", „(5)" usw.
- `caption_ref` am visuellen Element — Pointer auf das Caption-Element
- `extractor_version` am Element

Der ganze Phase-2-Teil ist nachgelagert **berechenbar** aus dem Phase-1-Output — es
braucht kein erneutes PDF-Parsing.

Sobald `document_rich.json` in Phase 2 produziert wird, muss die Stage-Sicherheit
um dieses Artefakt erweitert werden. Aktuell produziert die Pipeline
`content_list.json`, `segmentation.json`, Page-Bilder, Sidecars, Crops und
Stage-Marker.
