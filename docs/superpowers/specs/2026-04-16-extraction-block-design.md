# Extraction Block Design

## Ziel

Den Extraction Block als eigenständigen, isolierten Block aufbauen. PDF rein, strukturierter Output raus. Kein Embedding, kein Indexing, kein VectorDB. Die nachfolgenden Blöcke (Embedding, Indexing) konsumieren den Output und werden später separat gebaut.

## Langfristiger Kontext

Das System soll Claims in technischen PDFs verifizieren. Claims stützen sich auf Tabellenwerte, Formeln, Diagramme und Text. Dafür braucht der spätere Knowledge Graph saubere, semantisch reiche Extraktion mit erhaltener Dokumentstruktur und Beziehungen zwischen Elementen.

## Projektstruktur

```
techpdfparser/
├── extraction/          ← Extraction Block (jetzt)
├── embedding/           ← später
├── indexing/            ← später
```

- Jeder Block ist eigenständig, kein geteilter Code zwischen Blöcken.
- Duplizierung ist akzeptabel wenn dadurch Unabhängigkeit erhalten bleibt.
- Der bestehende Code unter `src/` wird vorher auf GitHub gepusht. Danach wird umstrukturiert.
- Tools bleiben per Config austauschbar (gleiches Prinzip wie bisher).

## Pipeline-Architektur

```
PDF
 |
 v
Segmentierung (Layout-Analyse)
 -> erkennt Regionen: Text, Tabelle, Formel, Bild, Diagramm, Tech. Zeichnung
 -> speichert eigenen Output (Regionen, Positionen, Typ-Klassifikation)
 -> speichert Seitenbilder und Element-Crops
 |
 v
Routing pro Region -> bester Extractor
 +-- Text-Region        -> OCR (z.B. OlmOCR)
 +-- Tabelle-Region     -> Tabellenparser (z.B. MinerU)
 +-- Formel-Region      -> Formelerkennung (z.B. PPFormulaNet)
 +-- Bild/Diagramm      -> Crop + Bildbeschreibung (z.B. Qwen2.5-VL)
 +-- Tech. Zeichnung    -> Crop + Bildbeschreibung
 |
 v
Merge in Reading Order
 -> Confidence-Filter (niedrige Confidence rausfiltern)
 -> Mehrseitige Elemente zusammenfuehren
 -> Captions zuordnen
 |
 v
Output: content_list.json + document_rich.json + Bilddateien
```

### Segmentierung

Die Segmentierung ist der erste Schritt und der Dirigent der Pipeline. Sie analysiert das Layout jeder Seite und entscheidet welcher Typ jede Region hat. Ihr Output wird separat gespeichert, damit er spaeter kontrolliert werden kann (z.B. durch Agents).

### Routing

Jede erkannte Region wird an genau einen Extractor geroutet, basierend auf dem Typ den die Segmentierung zugewiesen hat. Dadurch werden Dopplungen vermieden -- kein Extractor verarbeitet Regionen die nicht fuer ihn bestimmt sind.

### Confidence und Fehlerbehandlung

Jeder Extractor liefert einen Confidence-Wert pro Element. Der Merge-Schritt filtert Elemente mit zu niedriger Confidence heraus. Damit werden Segmentierungsfehler abgefangen: wenn die Segmentierung ein Bild faelschlich als Tabelle klassifiziert, erkennt der Tabellenparser keine Struktur und liefert niedrige Confidence.

Fallback-Logik (Region an alternativen Extractor weiterreichen) ist fuer spaeter vorgesehen, nicht fuer die erste Version.

## Output-Format

### Ordnerstruktur

```
outputs/<doc_id>/
+-- content_list.json
+-- document_rich.json
+-- segmentation.json        <- Segmentierungs-Rohdaten
+-- pages/
    +-- 1/
    |   +-- page.png
    |   +-- e003_diagram.png
    +-- 2/
    |   +-- page.png
    |   +-- e006_table.png
    |   +-- e007_formula.png
    +-- 3/
        +-- page.png
        +-- e010_drawing.png
        +-- e011_figure.png
```

- Jede Seite bekommt ein Ganzseitenbild (`page.png`).
- Jedes visuelle Element bekommt einen Crop basierend auf seiner Bounding Box.

### content_list.json

Flache Elementliste in Reading Order. Source of Truth fuer alle Inhalte. Wird spaeter vom Embedding Block konsumiert.

```json
{
  "doc_id": "a1b2c3",
  "source_file": "pruefbericht_2024.pdf",
  "total_pages": 3,
  "schema_version": "1.0",
  "segmentation_tool": "mineru25",

  "pages": [
    {
      "page": 1,
      "image_path": "pages/1/page.png",
      "element_ids": ["e001", "e002", "e003"]
    },
    {
      "page": 2,
      "image_path": "pages/2/page.png",
      "element_ids": ["e004", "e005", "e006", "e007"]
    },
    {
      "page": 3,
      "image_path": "pages/3/page.png",
      "element_ids": ["e008", "e009", "e010", "e011", "e012"]
    }
  ],

  "elements": [
    {
      "element_id": "e001",
      "type": "heading",
      "page": 1,
      "bbox": [80, 40, 900, 90],
      "reading_order_index": 0,
      "section_path": ["1. Einleitung"],
      "confidence": 0.98,
      "extractor": "olmocr2",
      "content": {
        "text": "1. Einleitung"
      }
    },
    {
      "element_id": "e002",
      "type": "text",
      "page": 1,
      "bbox": [80, 100, 900, 300],
      "reading_order_index": 1,
      "section_path": ["1. Einleitung"],
      "confidence": 0.95,
      "content": {
        "text": "Dieser Bericht dokumentiert die Druckpruefung des Behaelters DN-400 gemaess DIN EN 13445. Die Pruefung wurde am 12.03.2024 durchgefuehrt."
      }
    },
    {
      "element_id": "e003",
      "type": "diagram",
      "page": 1,
      "bbox": [120, 320, 850, 700],
      "reading_order_index": 2,
      "section_path": ["1. Einleitung"],
      "confidence": 0.91,
      "content": {
        "image_path": "pages/1/e003_diagram.png",
        "description": "Schematische Darstellung des Pruefaufbaus mit Drucksensor P1, Temperatursensor T1 und Ablassventil V1.",
        "caption": "Abbildung 1: Pruefaufbau fuer die Druckpruefung nach DIN EN 13445"
      }
    },
    {
      "element_id": "e004",
      "type": "heading",
      "page": 2,
      "bbox": [80, 40, 900, 90],
      "reading_order_index": 3,
      "section_path": ["2. Messergebnisse"],
      "confidence": 0.98,
      "content": {
        "text": "2. Messergebnisse"
      }
    },
    {
      "element_id": "e005",
      "type": "text",
      "page": 2,
      "bbox": [80, 100, 900, 200],
      "reading_order_index": 4,
      "section_path": ["2. Messergebnisse"],
      "confidence": 0.94,
      "content": {
        "text": "Tabelle 1 zeigt die gemessenen Druckwerte bei steigender Belastung. Der maximale Pruefdruck betrug 25 bar."
      }
    },
    {
      "element_id": "e006",
      "type": "table",
      "page": 2,
      "bbox": [100, 220, 880, 450],
      "reading_order_index": 5,
      "section_path": ["2. Messergebnisse"],
      "confidence": 0.93,
      "extractor": "mineru25",
      "content": {
        "markdown": "| Stufe | Druck [bar] | Temp [C] | Dehnung [mm] |\n|-------|------------|----------|-------------|\n| 1 | 5.0 | 21.3 | 0.02 |\n| 2 | 10.0 | 21.5 | 0.05 |\n| 3 | 15.0 | 21.8 | 0.09 |\n| 4 | 20.0 | 22.1 | 0.14 |\n| 5 | 25.0 | 22.6 | 0.21 |",
        "text": "Stufe 1: 5.0 bar, 21.3C, 0.02mm; Stufe 2: 10.0 bar, 21.5C, 0.05mm; Stufe 3: 15.0 bar, 21.8C, 0.09mm; Stufe 4: 20.0 bar, 22.1C, 0.14mm; Stufe 5: 25.0 bar, 22.6C, 0.21mm",
        "image_path": "pages/2/e006_table.png",
        "caption": "Tabelle 1: Gemessene Druckwerte bei steigender Belastung"
      }
    },
    {
      "element_id": "e007",
      "type": "formula",
      "page": 2,
      "bbox": [200, 470, 750, 530],
      "reading_order_index": 6,
      "section_path": ["2. Messergebnisse"],
      "confidence": 0.89,
      "content": {
        "latex": "\\sigma = \\frac{p \\cdot d}{2 \\cdot s} = \\frac{25 \\cdot 400}{2 \\cdot 12} = 416{,}7 \\text{ N/mm}^2",
        "text": "sigma = (p * d) / (2 * s) = (25 * 400) / (2 * 12) = 416,7 N/mm2",
        "image_path": "pages/2/e007_formula.png"
      }
    },
    {
      "element_id": "e008",
      "type": "heading",
      "page": 3,
      "bbox": [80, 40, 900, 90],
      "reading_order_index": 7,
      "section_path": ["3. Konstruktion"],
      "confidence": 0.98,
      "content": {
        "text": "3. Konstruktion"
      }
    },
    {
      "element_id": "e009",
      "type": "text",
      "page": 3,
      "bbox": [80, 100, 900, 200],
      "reading_order_index": 8,
      "section_path": ["3. Konstruktion"],
      "confidence": 0.95,
      "content": {
        "text": "Die technische Zeichnung in Abbildung 2 zeigt die Masse des Behaelters. Die Wandstaerke von 12 mm entspricht der Berechnung aus Abschnitt 2."
      }
    },
    {
      "element_id": "e010",
      "type": "technical_drawing",
      "page": 3,
      "bbox": [60, 210, 920, 550],
      "reading_order_index": 9,
      "section_path": ["3. Konstruktion"],
      "confidence": 0.90,
      "content": {
        "image_path": "pages/3/e010_drawing.png",
        "description": "Technische Zeichnung Behaelter DN-400, Wandstaerke 12mm, Flanschmasse DN-400 PN-40.",
        "caption": "Abbildung 2: Masszeichnung Druckbehaelter DN-400 PN-40"
      }
    },
    {
      "element_id": "e011",
      "type": "figure",
      "page": 3,
      "bbox": [100, 570, 500, 720],
      "reading_order_index": 10,
      "section_path": ["3. Konstruktion"],
      "confidence": 0.92,
      "content": {
        "image_path": "pages/3/e011_figure.png",
        "description": "Foto des Behaelters nach der Druckpruefung, keine sichtbaren Verformungen.",
        "caption": "Abbildung 3: Behaelter nach Pruefung"
      }
    },
    {
      "element_id": "e012",
      "type": "text",
      "page": 3,
      "bbox": [80, 730, 900, 800],
      "reading_order_index": 11,
      "section_path": ["3. Konstruktion"],
      "confidence": 0.96,
      "content": {
        "text": "Die Pruefung wurde bestanden. Keine plastische Verformung festgestellt."
      }
    }
  ]
}
```

### document_rich.json

Hierarchische Struktur und Relationen. Referenziert Elemente aus content_list.json per `$ref`. Dupliziert keine Inhalte.

```json
{
  "doc_id": "a1b2c3",
  "source_file": "pruefbericht_2024.pdf",
  "total_pages": 3,
  "schema_version": "1.0",
  "segmentation_tool": "mineru25",

  "sections": [
    {
      "heading": "1. Einleitung",
      "level": 1,
      "page_start": 1,
      "children": [
        { "$ref": "e002" },
        { "$ref": "e003" }
      ]
    },
    {
      "heading": "2. Messergebnisse",
      "level": 1,
      "page_start": 2,
      "children": [
        { "$ref": "e005" },
        { "$ref": "e006" },
        { "$ref": "e007" }
      ]
    },
    {
      "heading": "3. Konstruktion",
      "level": 1,
      "page_start": 3,
      "children": [
        { "$ref": "e009" },
        { "$ref": "e010" },
        { "$ref": "e011" },
        { "$ref": "e012" }
      ]
    }
  ],

  "relations": [
    { "from": "e005", "to": "e006", "type": "refers_to", "evidence": "Tabelle 1 zeigt" },
    { "from": "e009", "to": "e010", "type": "refers_to", "evidence": "Abbildung 2 zeigt" },
    { "from": "e009", "to": "e007", "type": "refers_to", "evidence": "Berechnung aus Abschnitt 2" }
  ]
}
```

### Element-Schema

Felder pro Element:

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| element_id | string | Eindeutige ID (Hash-basiert) |
| type | enum | text, heading, table, formula, figure, diagram, technical_drawing |
| page | int | Seitennummer |
| bbox | [x0,y0,x1,y1] | Bounding Box auf der Seite |
| reading_order_index | int | Position in der globalen Lesereihenfolge |
| section_path | string[] | Kapitelpfad, z.B. ["3. Konstruktion", "3.2 Masse"] |
| confidence | float | 0.0-1.0, vom Extractor geliefert |
| extractor | string | Name des Extractors der dieses Element erzeugt hat (z.B. "olmocr2", "mineru25") |
| content | object | Typ-abhaengige Inhalte (siehe unten) |

Content-Felder je nach Typ:

| Feld | Verwendet bei | Beschreibung |
|------|--------------|-------------|
| text | alle | Plaintext-Repraesentation |
| markdown | table | Tabelleninhalt als Markdown |
| latex | formula | Formel als LaTeX |
| image_path | table, formula, figure, diagram, technical_drawing | Pfad zum visuellen Crop |
| description | figure, diagram, technical_drawing | Generierte Bildbeschreibung |
| caption | table, figure, diagram, technical_drawing | Original-Caption aus dem PDF |

### Mehrseitige Elemente

Tabellen die ueber mehrere Seiten gehen werden zu einem Element zusammengefuehrt. Das resultierende Element hat die erste Seite als `page`-Wert. Die Bounding Boxes beider Seitenteile werden im Crop beruecksichtigt.

## Designentscheidungen

| Entscheidung | Begruendung |
|-------------|-------------|
| Zwei Output-Dateien statt einer | content_list ist Source of Truth fuer Inhalte; document_rich fuer Struktur/Relationen. Keine Duplizierung. |
| Segmentierung als erster Schritt | Vermeidet Dopplungen durch parallele Extractoren. Jede Region geht an genau einen Extractor. |
| Confidence pro Element | Faengt Segmentierungsfehler ab. Niedrige Confidence = Extractor konnte die Region nicht sinnvoll verarbeiten. |
| Crops fuer alle visuellen Elemente | Ermoeglicht spaeter multimodales Embedding ohne erneute Extraktion. |
| Kein geteilter Code zwischen Bloecken | Maximale Unabhaeingigkeit. Duplizierung ist billiger als falsche Kopplung. |
| Tools austauschbar per Config | Segmentierungstool, OCR, Tabellenparser etc. koennen gewechselt werden ohne Codeaenderung. |
| Kein Fallback bei v1 | Einfachheit. Confidence-Filterung reicht fuer den Anfang. |

## Nicht im Scope (kommt spaeter)

- Embedding Block (eigener Ordner `embedding/`)
- Indexing / VectorDB (eigener Ordner `indexing/`)
- Agent / Routing Layer
- Knowledge Graph
- Fallback-Logik bei Segmentierungsfehlern
- Kopf-/Fusszeilen-Erkennung
