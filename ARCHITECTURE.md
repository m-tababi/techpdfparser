# techpdfparser — Wie das Projekt aufgebaut ist und warum

---

## Das Kernproblem: Technische PDFs sind komplex

Eine technische PDF-Seite kann gleichzeitig enthalten:
- Fließtext in mehreren Spalten
- Mathematische Formeln
- Tabellen mit Messwerten
- Diagramme und Grafiken

Kein einziges KI-Tool ist für alle vier Dinge gleichzeitig das beste. Deswegen haben wir **drei getrennte Pipelines** gebaut, jede für einen anderen Inhalt.

Das zweite Problem: KI-Modelle veralten schnell. Was heute das beste OCR-Modell ist, ist in einem Jahr vielleicht überholt. Das System muss so gebaut sein, dass man ein Modell gegen ein anderes austauschen kann — ohne den Rest des Codes anzufassen.

---

## Die drei Pipelines — was jede macht

```
PDF-Datei
   │
   ├──► Visual Pipeline
   │    "Wie sieht die Seite als Bild aus?"
   │    → rendert Seite als PNG
   │    → berechnet visuelles Embedding (ColQwen2.5)
   │    → speichert in Qdrant (Collection: visual_pages)
   │
   ├──► Text Pipeline
   │    "Was steht da geschrieben?"
   │    → extrahiert Text via OCR (olmOCR-2)
   │    → teilt in kleine Abschnitte auf (Chunking)
   │    → berechnet Text-Embedding (BGE-M3)
   │    → speichert in Qdrant (Collection: text_chunks)
   │
   └──► Structured Pipeline
        "Gibt es Tabellen, Formeln, Abbildungen?"
        → erkennt Struktur (MinerU)
        → Formeln → LaTeX (PP-FormulaNet)
        → Abbildungen → Textbeschreibung (Qwen2.5-VL)
        → berechnet Embeddings (BGE-M3)
        → speichert in Qdrant (Collections: tables, formulas, figures)
```

Später bei einer Suchanfrage: alle drei Pipelines werden abgefragt und die Ergebnisse werden mit **Reciprocal Rank Fusion (RRF)** zusammengeführt.

---

## Projektstruktur — erklärt in einfachen Worten

```
src/
 ├── core/          Was das System TUN soll  (keine externen Tools)
 ├── adapters/      Wie es das mit konkreten Tools umgesetzt wird
 └── utils/         Kleine Hilfsfunktionen
```

Die wichtigste Regel: `core/` kennt weder PyMuPDF noch Qdrant noch ColQwen. Es weiß nur, **was** ein Renderer, ein Embedder oder eine Datenbank können muss — nicht **wie** sie es konkret tun.

### `src/core/models/` — Die Datenstrukturen

Diese Dateien definieren, wie Daten im System aussehen.

**`document.py`** — Metadaten zu einem Dokument:
```python
DocumentMeta(
    doc_id="a1b2c3d4",          # eindeutige ID, aus Dateipfad gehasht
    source_file="/data/paper.pdf",
    total_pages=12,
    file_size_bytes=2048000,
)

BoundingBox(x0=100, y0=200, x1=400, y1=350)  # wo auf der Seite ein Element sitzt
```

**`elements.py`** — Jedes extrahierte Objekt. Alle teilen gemeinsame Felder:

```python
# Gemeinsame Felder (jedes Element hat diese):
object_id        # eindeutige ID
doc_id           # aus welchem Dokument
source_file      # welche PDF-Datei
page_number      # auf welcher Seite (0-basiert)
tool_name        # welches Tool hat das extrahiert ("colqwen25", "mineru25", ...)
tool_version     # welche Version des Tools
bbox             # wo auf der Seite (oder None wenn unbekannt)
parent_id        # falls dieses Element aus einem größeren stammt
```

Dazu spezifische Felder je Typ:
```python
VisualPage   → image_path + embedding (Liste von Vektoren)
TextChunk    → content (der Text) + embedding (ein Vektor)
Table        → content (Markdown-Tabelle) + rows + headers + embedding
Formula      → latex (z.B. "E = mc^2") + content + embedding
Figure       → image_path + description + caption + embedding
```

**Warum so viele gemeinsame Felder?** Damit der Code, der Elemente in Qdrant speichert oder abruft, für alle fünf Typen gleich aussehen kann. Er muss nicht wissen ob es eine Formel oder ein Textabschnitt ist — er greift immer auf `element.doc_id`, `element.page_number` usw. zu.

**`results.py`** — Ergebnisstrukturen:
```python
RetrievalResult(
    element=<ein TextChunk oder VisualPage oder ...>,
    score=0.87,          # wie gut passt es zur Suchanfrage
    collection="text_chunks",
)

FusionResult(
    element=<...>,
    fused_score=0.73,    # kombinierter Score aus mehreren Pipelines
    source_scores={      # woher kam welcher Score
        "text_chunks": 0.87,
        "visual_pages": 0.65,
    }
)
```

---

### `src/core/interfaces/` — Was ein Tool können muss

Das ist der Schlüssel zur Austauschbarkeit. Statt zu sagen "benutze PyMuPDF", sagen wir "benutze irgendetwas das Seiten rendern kann".

In Python heißt das `Protocol`. Ein `Protocol` ist wie ein Vertrag:
> "Wenn deine Klasse diese Methoden hat, darfst du hier mitmachen."

**Beispiel — `renderer.py`:**
```python
class PageRenderer(Protocol):

    def render_page(self, pdf_path: Path, page_number: int) -> Image:
        # "Gib mir Seite X aus dieser PDF als Bild"
        ...

    def render_all(self, pdf_path: Path) -> list[Image]:
        # "Gib mir alle Seiten als Bilder"
        ...

    def page_count(self, pdf_path: Path) -> int:
        # "Wie viele Seiten hat diese PDF?"
        ...
```

Die Pipeline-Klasse (`VisualPipeline`) arbeitet **nur** mit diesem Vertrag. Sie weiß nicht, ob dahinter PyMuPDF, pdf2image oder etwas anderes steckt.

**Alle 12 Interfaces und was sie bedeuten:**

| Datei | Interface | Frage die es beantwortet |
|---|---|---|
| `renderer.py` | `PageRenderer` | Wie rendert man eine PDF-Seite? |
| `visual.py` | `VisualEmbedder` | Wie wandelt man ein Seitenbild in Vektoren um? |
| `extractor.py` | `TextExtractor` | Wie extrahiert man Text aus einer PDF? |
| `chunker.py` | `TextChunker` | Wie teilt man langen Text in kleine Stücke? |
| `embedder.py` | `TextEmbedder` | Wie wandelt man Text in einen Vektor um? |
| `parser.py` | `StructuredParser` | Wie findet man Tabellen, Formeln, Abbildungen? |
| `formula.py` | `FormulaExtractor` | Wie erkennt man Formeln und wandelt sie in LaTeX? |
| `figure.py` | `FigureDescriptor` | Wie beschreibt man eine Abbildung mit Text? |
| `indexer.py` | `IndexWriter` | Wie speichert man Embeddings in eine Datenbank? |
| `retriever.py` | `RetrievalEngine` | Wie sucht man in der Datenbank? |
| `fusion.py` | `FusionEngine` | Wie kombiniert man Ergebnisse mehrerer Suchen? |
| `benchmark.py` | `BenchmarkRunner` | Wie misst man die Performance einer Pipeline? |

---

### `src/core/config.py` — Die Konfiguration

Die gesamte Tool-Auswahl steht in einer YAML-Datei:

```yaml
# config.yaml
pipelines:
  visual:
    renderer: "pymupdf"      # ← dieser Name bestimmt welches Tool benutzt wird
    embedder: "colqwen25"
  text:
    extractor: "olmocr2"
    embedder: "bge_m3"
```

`load_config("config.yaml")` liest diese Datei und gibt ein Python-Objekt zurück:

```python
cfg = load_config("config.yaml")
cfg.pipelines.visual.embedder   # → "colqwen25"
cfg.pipelines.text.extractor    # → "olmocr2"
```

Pydantic validiert dabei automatisch: falsche Typen, fehlende Pflichtfelder — alles wird sofort als Fehler gemeldet. Keine unerwarteten Abstürze später im Code.

---

### `src/core/registry.py` — Das Telefonbuch der Adapter

Das Registry löst das konkrete Problem: Der Name `"colqwen25"` aus der Config muss irgendwie zur Klasse `ColQwen25Embedder` werden.

**Wie Adapter sich registrieren** (in jeder Adapter-Datei):
```python
@register_visual_embedder("colqwen25")   # ← "colqwen25" = der Name in der Config
class ColQwen25Embedder:
    def embed_page(self, image): ...
    def embed_query(self, query): ...
```

Der Dekorator `@register_visual_embedder("colqwen25")` trägt die Klasse in ein internes Dict ein:
```python
_VISUAL_EMBEDDERS = {
    "colqwen25": ColQwen25Embedder,
    "colpali": ColPaliEmbedder,   # wenn jemand das hinzufügt
    ...
}
```

**Wie die Pipeline ein Tool abruft:**
```python
embedder_name = cfg.pipelines.visual.embedder   # → "colqwen25"
adapter_cfg = get_adapter_config(cfg, "colqwen25")  # → {"device": "cuda", ...}
embedder = get_visual_embedder(embedder_name, **adapter_cfg)
# → gibt ColQwen25Embedder(device="cuda") zurück
```

**Was passiert beim Tool-Wechsel:**
```yaml
# config.yaml
pipelines:
  visual:
    embedder: "colpali"   # ← geändert
```
→ `get_visual_embedder("colpali")` → gibt `ColPaliEmbedder` zurück.
Kein anderer Code ändert sich.

---

### `src/core/pipelines/` — Der Dirigent

Die Pipeline-Klassen bekommen ihre Adapter von außen übergeben (Dependency Injection) und rufen sie in der richtigen Reihenfolge auf.

**`visual.py` — VisualPipeline in Kurzform:**
```python
class VisualPipeline:
    def __init__(self, renderer, embedder, index_writer, storage, config):
        self.renderer = renderer        # ← konkret: PyMuPDFRenderer
        self.embedder = embedder        # ← konkret: ColQwen25Embedder
        self.index_writer = index_writer  # ← konkret: QdrantIndexWriter

    def run(self, pdf_path, doc_meta):
        # Schritt 1: Alle Seiten rendern
        images = self.renderer.render_all(pdf_path)

        # Schritt 2: Jede Seite speichern + Embedding berechnen
        for page_num, image in enumerate(images):
            image.save(...)              # PNG auf Disk
            embedding = self.embedder.embed_page(image)  # Vektoren berechnen
            pages.append(VisualPage(...))

        # Schritt 3: Alles in Qdrant schreiben
        self.index_writer.upsert_visual(self.config.collection, pages)
```

Die Pipeline weiß nicht wie PyMuPDF intern funktioniert. Sie ruft nur `.render_all()` auf. Wenn man PyMuPDF gegen pdf2image austauscht, funktioniert die Pipeline identisch weiter.

---

### `src/adapters/` — Die konkreten Werkzeuge

Hier steckt der eigentliche ML-Code. Jeder Adapter implementiert genau ein Interface.

**Wichtigstes Design-Merkmal: Lazy Loading**

Modelle werden nicht beim Import geladen, sondern erst beim ersten echten Aufruf:

```python
class ColQwen25Embedder:
    def __init__(self, model_name, device):
        self._model = None    # noch nicht geladen!

    def _load(self):
        if self._model is not None:
            return   # schon geladen, nichts tun
        from colpali_engine.models import ColQwen2_5
        self._model = ColQwen2_5.from_pretrained(self._model_name)

    def embed_page(self, image):
        self._load()   # jetzt erst laden, wenn wirklich gebraucht
        ...
```

**Warum?** Weil man vielleicht nur die Text-Pipeline laufen lassen will, ohne ColQwen zu laden. Das spart mehrere Minuten Wartezeit und GBs RAM.

**Wie ein Adapter aussieht — `pymupdf.py` als einfachstes Beispiel:**
```python
@register_renderer("pymupdf")      # Registrierung: "pymupdf" → diese Klasse
class PyMuPDFRenderer:

    TOOL_NAME = "pymupdf"
    TOOL_VERSION = "1.24"

    def __init__(self, dpi=150):
        self._dpi = dpi
        # fitz = der interne Name von PyMuPDF
        import fitz
        self._fitz = fitz

    def render_page(self, pdf_path, page_number):
        with self._fitz.open(str(pdf_path)) as doc:
            page = doc[page_number]
            # Matrix skaliert von PDF-Einheiten (72) auf gewünschte DPI
            mat = self._fitz.Matrix(self._dpi / 72, self._dpi / 72)
            pixmap = page.get_pixmap(matrix=mat, alpha=False)
            return PIL.Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

    def render_all(self, pdf_path):
        count = self.page_count(pdf_path)
        return [self.render_page(pdf_path, i) for i in range(count)]
```

**Übersicht aller Adapter:**

| Ordner | Datei | Tool | Was es macht |
|---|---|---|---|
| `renderers/` | `pymupdf.py` | PyMuPDF | PDF-Seite → PNG |
| `visual/` | `colqwen25.py` | ColQwen2.5 | PNG → Patch-Vektoren (Multi-Vektor) |
| `ocr/` | `olmocr2.py` | olmOCR-2 | PNG → Markdown-Text (OCR) |
| `parsers/` | `mineru25.py` | MinerU 2.5 | PDF → Tabellen + Formeln + Abbildungen |
| `formula/` | `ppformulanet.py` | PP-FormulaNet | Formel-Bild → LaTeX |
| `figures/` | `qwen25vl.py` | Qwen2.5-VL | Abbildungs-Bild → Textbeschreibung |
| `vectordb/` | `qdrant.py` | Qdrant | Embeddings speichern + suchen |
| `embedders/` | `bge_m3.py` | BGE-M3 | Text → Vektor (1024 Dimensionen) |
| `chunkers/` | `fixed_size.py` | — | Langen Text in kleine Stücke teilen |
| `fusion/` | `rrf.py` | — | Mehrere Suchergebnislisten zusammenführen |

---

### `src/utils/` — Kleine Helfer

**`ids.py` — Warum stabile IDs wichtig sind**

Wenn wir dieselbe PDF zweimal verarbeiten, soll dasselbe Element dieselbe ID bekommen. So kann Qdrant es einfach überschreiben (upsert) statt es doppelt zu speichern.

```python
generate_element_id("doc_abc", page=0, "visual_page", "colqwen25")
# → immer: "3f7a2b1c9e4d8a06"

# Zusammensetzung: SHA256 von "doc_abc:0:visual_page:colqwen25:0"
# dann die ersten 16 Zeichen des Hex-Strings
```

Jede Kombination aus Dokument + Seite + Typ + Tool ergibt eine andere ID. So können Ergebnisse von ColQwen und ColPali für dieselbe Seite gleichzeitig existieren.

**`timing.py` — Laufzeitmessung**

```python
with timed("embed_batch") as t:
    embeddings = model.encode(texts)

# t.elapsed_seconds → z.B. 2.37
```

Alle Pipelines messen automatisch wie lange jeder Schritt dauert. Das ist die Grundlage für spätere Benchmarks.

**`storage.py` — Versionierte Ausgabeverzeichnisse**

```python
storage.run_dir("visual", "colqwen25")
# → outputs/visual/visual_colqwen25_20260409_204300/
```

Jeder Pipeline-Lauf bekommt einen eigenen Ordner mit Zeitstempel. Läuft man die Pipeline zweimal (z.B. mit zwei verschiedenen Modellen), überschreibt der zweite Lauf den ersten nicht. Das macht Vergleiche einfach.

**`logging.py` — Einheitliches Logging**

```
2026-04-09T20:43:00 | INFO     | techpdfparser.pipelines.visual | Visual pipeline start | doc=a1b2c3d4
2026-04-09T20:43:02 | INFO     | techpdfparser.pipelines.visual | Rendered and embedded 12 pages in 1.87s
```

Alle Log-Meldungen folgen demselben Format. So kann man sie später mit Tools wie grep oder Loki auswerten.

---

## Wie hängt alles zusammen — ein kompletter Durchlauf

```
1. config.yaml laden
   cfg = load_config("config.yaml")

2. Adapter instanziieren (per Registry)
   renderer     = get_renderer("pymupdf", dpi=150)
   embedder     = get_visual_embedder("colqwen25", device="cuda")
   index_writer = get_index_writer("qdrant", host="localhost", port=6333)

3. Pipeline zusammenbauen
   pipeline = VisualPipeline(renderer, embedder, index_writer, storage, cfg.pipelines.visual)

4. Pipeline laufen lassen
   pages = pipeline.run(Path("paper.pdf"), doc_meta)
   # → intern:
   #    images = renderer.render_all(pdf)         # PyMuPDF rendert
   #    embedding = embedder.embed_page(image)    # ColQwen2.5 berechnet
   #    index_writer.upsert_visual(...)           # Qdrant speichert

5. Ergebnis: 12 VisualPage-Objekte in Qdrant gespeichert
```

---

## Wie man ein Tool austauscht — konkret

Angenommen wir wollen **ColPali statt ColQwen2.5** testen.

**Schritt 1:** Neue Datei anlegen: `src/adapters/visual/colpali.py`
```python
@register_visual_embedder("colpali")   # ← neuer Name
class ColPaliEmbedder:
    # ... ColPali-spezifischer Code ...
    def embed_page(self, image): ...
    def embed_query(self, query): ...
```

**Schritt 2:** In `src/adapters/visual/__init__.py` importieren:
```python
from . import colqwen25
from . import colpali   # ← neu
```

**Schritt 3:** In `config.yaml` ändern:
```yaml
pipelines:
  visual:
    embedder: "colpali"   # ← war "colqwen25"
```

**Fertig.** Keine andere Datei wird angefasst. Die Pipeline, die Datenmodelle, die Qdrant-Anbindung — alles bleibt unverändert.

---

## Die Tests — warum sie ohne GPU laufen

Alle 48 Tests testen die Logik, nicht die Modelle:

- `test_models.py` — Können Pydantic-Objekte serialisiert und wieder deserialisiert werden?
- `test_config.py` — Wird YAML korrekt geladen und validiert?
- `test_ids.py` — Liefert dieselbe Eingabe immer dieselbe ID?
- `test_chunker.py` — Wird Text korrekt in Stücke geteilt, mit richtigem Overlap?
- `test_rrf.py` — Berechnet RRF die richtigen Fusion-Scores?
- `test_pipelines.py` — Ruft die Pipeline ihre Adapter in der richtigen Reihenfolge auf?

Für `test_pipelines.py` werden echte Adapter durch **Mock-Objekte** ersetzt:
```python
embedder = MagicMock()
embedder.embed_page.return_value = [[0.1, 0.2]]  # gibt immer diesen Dummy-Vektor zurück

pipeline = VisualPipeline(renderer, embedder, index_writer, storage, config)
pipeline.run(...)

# Wir prüfen: wurde embed_page überhaupt aufgerufen?
embedder.embed_page.assert_called()
```

So können wir sicherstellen, dass die Pipeline korrekt orchestriert — ohne ColQwen2.5 laden zu müssen.

---

## Kurzfassung für schnelles Nachschlagen

```
src/core/models/      → Wie Daten aussehen (Pydantic-Klassen)
src/core/interfaces/  → Was ein Tool KÖNNEN muss (Protocol-Klassen)
src/core/config.py    → Welches Tool BENUTZT wird (aus config.yaml)
src/core/registry.py  → Tool-Name → Tool-Klasse (das "Telefonbuch")
src/core/pipelines/   → In welcher REIHENFOLGE wird was aufgerufen
src/adapters/         → Wie ein Tool konkret FUNKTIONIERT (ML-Code)
src/utils/            → IDs, Timing, Speicherpfade, Logging
tests/                → Prüft Logik ohne GPU/Modelle (Mocks)
config.example.yaml   → Alle konfigurierbaren Tool-Namen auf einen Blick
```
