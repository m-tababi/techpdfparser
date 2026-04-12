# Architektur — techpdfparser

> Ziel dieser Datei: dich so durch den Code führen, dass du danach jede
> Änderung am System mit Überblick machen kannst — nicht "wie ein User", der
> Tools konfiguriert, sondern "wie jemand, der den Code selbst geschrieben
> hat". Jedes Kapitel erklärt **was** passiert, **warum** es so und nicht
> anders gebaut ist, und zeigt die **Schlüsselstellen** im Code mit
> Dateipfad:Zeile, damit du direkt reinspringen kannst.

## Inhaltsverzeichnis

1. [Big Picture — was macht das Tool überhaupt?](#1-big-picture)
2. [Verzeichnisstruktur & Dependency-Richtung](#2-verzeichnisstruktur--dependency-richtung)
3. [Modularität: das Adapter-Registry-System](#3-modularität-das-adapter-registry-system)
4. [Datenmodelle: `BaseElement` und seine 5 Subklassen](#4-datenmodelle)
5. [Stabile IDs: warum wir SHA256 nehmen](#5-stabile-ids)
6. [Die 3 Pipelines — Visual, Text, Structured](#6-die-3-pipelines)
7. [Section-Awareness: Struktur als Querschnitts-Information](#7-section-awareness)
8. [Storage-Layout: wie Daten auf der Disk aussehen](#8-storage-layout)
9. [Qdrant-Indexing: Vektoren und Payloads](#9-qdrant-indexing)
10. [Retrieval & Fusion: die Ergebnisse wieder zusammenführen](#10-retrieval--fusion)
11. [Config-System: Tool-Swap per YAML](#11-config-system)
12. [End-to-End-Walkthrough: was bei `python -m src ingest test.pdf` passiert](#12-end-to-end-walkthrough)
13. [Tests: was wird wo getestet](#13-tests)

---

## 1. Big Picture

Das Tool nimmt ein PDF und macht daraus **durchsuchbare, strukturierte
Daten** in einer Vektordatenbank (Qdrant). Der Clou ist, dass nicht *ein*
Parser alles macht, sondern **drei unabhängige Pipelines parallel** laufen,
die jeweils eine andere Sichtweise auf das Dokument einnehmen:

```
┌─────────────────────────────────────────────────────────────────┐
│                            PDF                                  │
└─────────────────────────────────────────────────────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────────┐
│  Visual      │       │  Text        │       │  Structured      │
│  Pipeline    │       │  Pipeline    │       │  Pipeline        │
│              │       │              │       │                  │
│ Seiten als   │       │ Text extrah. │       │ Tables, Formeln, │
│ Bilder →     │       │ → chunken →  │       │ Figures detekt.  │
│ Vision-      │       │ embed        │       │ → embed          │
│ Embedding    │       │              │       │                  │
└──────┬───────┘       └──────┬───────┘       └─────────┬────────┘
       │                      │                         │
       ▼                      ▼                         ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────────┐
│visual_pages  │       │ text_chunks  │       │ tables           │
│ collection   │       │ collection   │       │ formulas         │
│              │       │              │       │ figures          │
└──────────────┘       └──────────────┘       └──────────────────┘
       │                      │                         │
       └──────────────────────┴─────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  UnifiedRetriever       │
                 │  (eine Query → alles)   │
                 │  + RRF-Fusion           │
                 └─────────────────────────┘
```

**Warum drei Pipelines statt einer?** Weil die Stärken komplementär sind:

- **Visual** fängt Layout, Diagramme, handschriftliche Notizen, Logos, alles
  was Text-Extraktion nicht sieht. ColQwen2.5/ColPali und CLIP arbeiten auf
  der Pixel-Ebene.
- **Text** fängt alles, was sich als Fließtext ausdrücken lässt —
  Paragraphs, Listen, Überschriften. Schneller und präziser als Visual, aber
  blind für nicht-textuelle Bedeutung.
- **Structured** zieht separat die *Objekte* raus, die man einzeln
  referenzieren können muss: Tabellen, Formeln, Figures. Die haben eigene
  Repräsentationen (Markdown-Tabelle, LaTeX, Bild + VLM-Beschreibung).

Am Ende, beim Retrieval, werden die 5 resultierenden Collections (visual,
text, tables, formulas, figures) wieder zusammengeführt mittels **Reciprocal
Rank Fusion**, sodass eine einzige Query alles gleichzeitig trifft.

---

## 2. Verzeichnisstruktur & Dependency-Richtung

```
src/
├── __main__.py                   # CLI-Entrypoint (python -m src ingest …)
├── core/
│   ├── config.py                 # YAML-Schema (Pydantic)
│   ├── registry.py               # Adapter-Registry (Kernstück der Modularität)
│   ├── retrieval.py              # UnifiedRetriever (Fusion über alle Collections)
│   ├── models/
│   │   ├── document.py           # DocumentMeta, BoundingBox
│   │   ├── elements.py           # BaseElement + 5 Subklassen
│   │   └── results.py            # RetrievalResult, FusionResult, Benchmark
│   ├── interfaces/               # Protocol-Klassen — die "Verträge"
│   │   ├── renderer.py
│   │   ├── visual.py             # VisualEmbedder
│   │   ├── extractor.py          # TextExtractor
│   │   ├── chunker.py
│   │   ├── embedder.py           # TextEmbedder
│   │   ├── parser.py             # StructuredParser
│   │   ├── formula.py
│   │   ├── figure.py
│   │   ├── indexer.py            # IndexWriter
│   │   ├── retriever.py          # RetrievalEngine
│   │   └── fusion.py             # FusionEngine
│   └── pipelines/
│       ├── visual.py             # VisualPipeline
│       ├── text.py               # TextPipeline
│       └── structured.py         # StructuredPipeline
├── adapters/                     # Konkrete Implementierungen der Protokolle
│   ├── __init__.py               # Kaskaden-Import triggert alle @register_*
│   ├── renderers/                # PyMuPDF
│   ├── visual/                   # ColQwen2.5, CLIP
│   ├── ocr/                      # olmOCR2, pymupdf_text, pymupdf_structured
│   ├── chunkers/                 # fixed_size, section_aware
│   ├── embedders/                # BGE-M3, MiniLM
│   ├── parsers/                  # MinerU2.5, pdfplumber
│   ├── formula/                  # PP-FormulaNet, pix2tex
│   ├── figures/                  # Qwen2.5-VL, moondream, noop
│   ├── vectordb/                 # Qdrant (Writer + Retriever)
│   └── fusion/                   # RRF, min-max score_norm
└── utils/
    ├── ids.py                    # SHA256-basierte stabile IDs
    ├── jsonl.py                  # JSONL-Writer (streaming)
    ├── manifest.py               # ManifestBuilder (Run-Metadaten)
    ├── sections.py               # SectionMarker + TOC/Font-Detection
    ├── storage.py                # StorageManager (Layout auf Disk)
    ├── timing.py                 # timed() Context-Manager
    └── logging.py                # strukturiertes Logging
```

**Die Dependency-Richtung ist strikt einbahnig:**

```
utils ─▶ core/models ─▶ core/interfaces ─▶ core/pipelines ─▶ adapters
```

Das heißt in Klartext:

- **`utils/`** darf nichts aus `core/` oder `adapters/` importieren → reine
  Helfer, isoliert testbar.
- **`core/models/`** sind reine Pydantic-Datenklassen — kennen keine
  Pipelines und keine konkreten Tools.
- **`core/interfaces/`** sind `Protocol`-Klassen (Duck-Typing-Verträge) —
  kennen nur Models, keine konkreten Implementierungen.
- **`core/pipelines/`** kennen Interfaces und Models, aber **keine
  konkreten Adapter**. Sie bekommen die Adapter per Constructor-Injection.
- **`adapters/`** dürfen `core/` importieren, um die Protokolle zu erfüllen
  und sich im Registry zu registrieren. Andersrum geht nicht.

Das garantiert zwei Dinge:

1. **Kein Zirkel-Import**, weil jede Schicht nur "nach unten" schaut.
2. **Pipelines sind generisch** — sie wissen nicht, ob der Renderer
   PyMuPDF oder pdf2image ist. Das ist das Fundament der Tool-Swap-Fähigkeit.

---

## 3. Modularität: das Adapter-Registry-System

Das ist der wichtigste Teil. Alles andere hängt davon ab.

### Das Problem

Wir wollen Tools (Modelle, Parser, Renderer) **per Config** austauschen, ohne
Code zu ändern. Naiv wäre: eine riesige `if`-Kette in der Pipeline, die je
nach Config-String die richtige Klasse instanziiert. Das schlägt aus zwei
Gründen fehl:

- Jeder neue Adapter erfordert Änderung am Pipeline-Code → enge Kopplung.
- Die Pipeline muss alle Adapter kennen → zirkuläre Imports.

### Die Lösung: Registry + Decorator

`src/core/registry.py` (78 Zeilen) baut mit einer Factory-Funktion für jeden
Typ ein Register-Decorator und eine Lookup-Funktion:

```python
# src/core/registry.py:29
def _make_register(registry: dict[str, type]):
    def register(name: str):
        def decorator(cls: type) -> type:
            registry[name] = cls
            return cls
        return decorator
    return register

_TEXT_EXTRACTORS: dict[str, type] = {}
register_text_extractor = _make_register(_TEXT_EXTRACTORS)
```

Jeder Adapter trägt sich selbst ein — in **seiner eigenen Datei**, am Klassen-
kopf:

```python
# src/adapters/ocr/pymupdf_structured.py:18
@register_text_extractor("pymupdf_structured")
class PyMuPDFStructuredExtractor:
    ...
```

Die Pipeline wiederum kennt nur die Lookup-Funktion:

```python
# src/__main__.py:80
extractor=get_text_extractor(tc.extractor, **get_adapter_config(cfg, tc.extractor))
```

`tc.extractor` ist der String aus der `config.yaml`
(`extractor: "pymupdf_structured"`), und `get_adapter_config(...)` holt sich
die adapterspezifischen kwargs. So landet *genau ein String aus dem YAML*
im Dictionary-Lookup und wird zu einer instanziierten Klasse.

### Aber wann werden die `@register_*`-Decorators ausgeführt?

Das ist die subtile Stelle, die leicht bricht: **Decorators laufen erst,
wenn das Modul importiert wird**. Wenn niemand `pymupdf_structured.py`
importiert, ist `"pymupdf_structured"` nicht im Registry — und
`get_text_extractor("pymupdf_structured")` wirft `KeyError`.

Deshalb gibt es **Kaskaden-Imports** in jedem `__init__.py`:

```python
# src/adapters/__init__.py
from . import chunkers, embedders, figures, formula, fusion, ocr, \
              parsers, renderers, vectordb, visual  # noqa: F401

# src/adapters/ocr/__init__.py
from . import olmocr2         # noqa: F401
from . import pymupdf_structured  # noqa: F401
from . import pymupdf_text    # noqa: F401
```

Und der CLI-Entrypoint importiert das Dach-Package **bevor** er den
Registry-Lookup macht:

```python
# src/__main__.py:8
import src.adapters  # noqa: F401 — triggers all @register_* decorators
```

**Praktisch heißt das:** wenn du einen neuen Adapter hinzufügst, reicht es
nicht, die Datei mit dem Decorator anzulegen. Du musst die Datei auch im
`__init__.py` der passenden Unterkategorie importieren. Vergisst du das,
laufen die Tests (die direkt den Pfad importieren), aber der CLI-Lauf knallt.

### Was das Registry dir einsparen soll

Einen neuen Text-Extractor hinzufügen, z. B. `docling_text`:

1. Neue Datei `src/adapters/ocr/docling_text.py` anlegen
2. Klasse mit `@register_text_extractor("docling_text")` dekorieren
3. Ein `from . import docling_text` in `src/adapters/ocr/__init__.py`
4. In `config.yaml`: `extractor: "docling_text"`

**Keine andere Datei wird angefasst.** Weder Pipeline, noch Models, noch
andere Adapter. Das ist der ganze Punkt.

---

## 4. Datenmodelle

`src/core/models/elements.py` (91 Zeilen) definiert das Herzstück der
Datenstruktur.

### Die Vererbungs-Hierarchie

```
                     BaseElement
                     │   (Pydantic BaseModel, gemeinsame Felder)
                     │
        ┌────────────┼────────────┬────────────┬──────────┐
        ▼            ▼            ▼            ▼          ▼
   VisualPage    TextChunk      Table       Formula     Figure
```

### Warum eine gemeinsame Basisklasse?

Weil **Writer und Retriever generisch über jeden Element-Typ iterieren
können** — mit `_base_payload(element)` in `qdrant.py:35` baut der
Indexer dieselben Felder (`doc_id`, `source_file`, `page_number`,
`object_type`, `tool_name`, `bbox`) für jedes Element, egal welcher Typ.

### Die Felder von `BaseElement`

```python
# src/core/models/elements.py:11
class BaseElement(BaseModel):
    object_id: str                       # 16-char SHA256, stabil über Re-Runs
    doc_id: str                          # 16-char SHA256 des PDF-Pfads
    source_file: str                     # Originalpfad zum PDF
    page_number: int                     # 0-basiert
    object_type: str                     # Discriminator-Feld (Pydantic union)
    bbox: BoundingBox | None = None      # Position auf der Seite
    tool_name: str                       # Welches Tool extrahiert hat
    tool_version: str
    extraction_timestamp: datetime       # Wann
    raw_output_path: str | None = None   # Pfad zur Roh-Datei (Bild, .md, …)
    parent_id: str | None = None         # Beim Chunking: parent-block
    child_ids: list[str] = []            # Umgekehrte Richtung

    # --- Seit Section-Awareness (siehe Kapitel 7) ---
    section_title: str | None = None
    section_path: list[str] = []         # z. B. ["2 Methods", "2.3 Data"]
    heading_level: int | None = None
```

### Die 5 Subklassen — jede mit ihrer Spezial-Payload

| Klasse | Zusatzfelder | Embedding-Typ |
|---|---|---|
| `VisualPage` | `image_path`, `embedding: list[list[float]]` | **Multi-Vector** (Patch-Vektoren) oder Single wrapped |
| `TextChunk` | `content`, `char_start`, `char_end`, `embedding: list[float]` | Single-Vector (Dense) |
| `Table` | `content` (Markdown), `rows`, `headers` | Single-Vector (auf Markdown) |
| `Formula` | `latex=""`, `content`, `image_path` | Single-Vector (auf LaTeX oder Plaintext) |
| `Figure` | `image_path`, `description`, `caption` | Single-Vector (auf Description/Caption) |

`Formula.latex` hat den Default `""`, weil manche Parser (z. B. pdfplumber)
eine Region finden aber kein LaTeX extrahieren können — ohne Default würde
Pydantic crashen. Das war ein Bug-Fix aus dem Section-Awareness-Work.

### Die Discriminated Union — warum das wichtig ist

```python
# src/core/models/elements.py:86
ExtractedElement = Annotated[
    Union[VisualPage, TextChunk, Table, Formula, Figure],
    Field(discriminator="object_type"),
]
```

Das sagt Pydantic: "Wenn du ein Dict deserialisierst und ein Feld
`object_type: "table"` drin ist, dann ist es eine `Table`, nicht eine
andere Klasse." Warum brauchen wir das?

Weil wir beim Retrieval den **Qdrant-Payload zurück in ein Pydantic-Modell**
wandeln müssen (`_payload_to_element` in `qdrant.py:266`). Wenn der
Payload `"object_type": "formula"` hat, bekommen wir automatisch eine
`Formula`, und der UnifiedRetriever kann trotz gemischter Ergebnistypen
einheitlich über die Liste iterieren.

---

## 5. Stabile IDs

`src/utils/ids.py` (23 Zeilen):

```python
def generate_element_id(doc_id, page_number, object_type, tool_name, sequence=0):
    raw = f"{doc_id}:{page_number}:{object_type}:{tool_name}:{sequence}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def generate_doc_id(source_file):
    return hashlib.sha256(source_file.encode()).hexdigest()[:16]
```

**Warum deterministisch und nicht UUID v4?**

Wenn du das gleiche PDF zweimal ingestierst, sollen die IDs der
extrahierten Elemente **identisch** sein. Qdrant-Upserts sind dann
idempotent: gleicher ID-Schlüssel → derselbe Point wird überschrieben, nicht
dupliziert. Bei UUID v4 hättest du nach zwei Runs zwei Kopien jedes
Elements.

Deshalb ist der ID-Input strukturiert: alle Felder, die ein Element
eindeutig identifizieren (Dokument + Seite + Typ + welches Tool + welche
Sequenznummer auf der Seite). Wenn du das Tool wechselst (z. B. von
`pymupdf_text` auf `pymupdf_structured`), ändern sich die IDs — das ist
gewollt, weil sonst zwei Runs mit verschiedenen Tools sich gegenseitig
überschreiben würden.

**Warum nur 16 Hex-Zeichen?**

`16*4 = 64 Bit`. Das reicht für Kollisions-Sicherheit in unserer
Größenordnung (Millionen Elemente, nicht Milliarden), und die kurzen IDs
sind lesbarer in Logs.

**Kleiner Sonderfall:** Qdrants In-Memory-Client akzeptiert keine
Hex-Strings als Point-IDs, nur UUIDs. Deshalb wird in `qdrant.py:25` die
16-Char Hex auf 32 Chars gepolstert und als UUID formatiert — der Output
bleibt deterministisch:

```python
def _to_point_id(hex_id: str) -> str:
    import uuid
    return str(uuid.UUID(hex=hex_id.ljust(32, "0")))
```

---

## 6. Die 3 Pipelines

Jede Pipeline-Klasse folgt demselben Aufbau:

1. **Constructor**: nimmt alle Adapter per Injection + StorageManager + Config
2. **`run(pdf_path, doc_meta)`**: orchestriert den Fluss, ruft Adapter, schreibt Outputs
3. **private `_*` Helper** für einzelne Schritte

Alle drei schreiben in dasselbe Run-Dir-Schema und benutzen denselben
`ManifestBuilder`, damit man nachher pro Run nachvollziehen kann, welches
Tool wie lange gebraucht hat und wie viele Elemente rauskamen.

### 6.1 VisualPipeline

`src/core/pipelines/visual.py`

```
┌────────┐     ┌──────────┐      ┌──────────────┐     ┌─────────────────┐
│ PDF    │ ──▶ │ Renderer │ ──▶  │ Visual       │ ──▶ │ IndexWriter     │
│        │     │ (PyMuPDF)│      │ Embedder     │     │ (Qdrant multi-  │
│        │     │ 150 DPI  │      │ (ColQwen2.5/ │     │ vector)         │
│        │     │          │      │  CLIP)       │     │                 │
└────────┘     └──────────┘      └──────────────┘     └─────────────────┘
                     │                    │                    │
                     ▼                    ▼                    ▼
              page_X.png auf         embedding:           visual_pages
              disk schreiben     list[list[float]]        collection
```

Jede Seite wird:
1. Mit `renderer.render_all(pdf_path)` zu einem `PIL.Image` gerendert
2. Als PNG unter `run_dir/pages/p0000.png` persistiert (für spätere
   Referenz, z. B. wenn der Retriever einen Treffer zeigen will)
3. Mit `embedder.embed_page(image)` in einen Vektor (oder Multi-Vektor)
   verwandelt
4. Als `VisualPage`-Objekt mit stabilem ID erzeugt
5. Nach allen Seiten: einmalig in Qdrant hochgeladen

**Warum `is_multi_vector`?** Late-Interaction-Modelle wie ColQwen2.5 geben
pro Bild nicht einen, sondern ~N Vektoren (einen pro Patch) zurück. Qdrant
hat dafür einen speziellen `MaxSim`-Comparator, der die
Bild-Query-Ähnlichkeit über alle Patch-Paare maximiert. Single-Vector-
Modelle wie CLIP wickeln ihren einen Vektor in `[[v0, v1, …]]` ein, sodass
das Datenformat uniform bleibt. In der Collection-Config wird der
`is_multi_vector`-Flag gelesen, um zu wissen, ob eine normale Cosine- oder
MaxSim-Collection erstellt wird (`qdrant.py:79`).

### 6.2 TextPipeline

`src/core/pipelines/text.py`

```
┌────────┐    ┌────────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
│ PDF    │───▶│ TextExtrctr│───▶│ Chunker  │───▶│ TextEmbed- │───▶│ Qdrant   │
│        │    │            │    │          │    │ der        │    │ upsert   │
│        │    │ pymupdf_   │    │ section_ │    │            │    │          │
│        │    │ structured │    │ aware    │    │ minilm /   │    │          │
│        │    │ (mit Font- │    │ (respekt.│    │ BGE-M3     │    │          │
│        │    │ Metadaten) │    │ Sections)│    │            │    │          │
└────────┘    └────────────┘    └──────────┘    └────────────┘    └──────────┘
                    │                 │                                  │
                    │                 │                                  ▼
                    ▼                 ▼                           text_chunks
              raw_blocks.jsonl  chunks.jsonl                       collection
              sections.json*
                (*falls Extractor Sections liefert)
```

Die Pipeline hat **einen zusätzlichen Persistence-Schritt** gegenüber
Visual: die rohen Extractor-Outputs (`raw_blocks.jsonl`) werden **vor** dem
Chunking geschrieben. So kannst du später den Chunker alleine auf neuen
Blöcken laufen lassen, ohne die (teure) OCR wiederholen zu müssen.

**Die `sections.json` ist ein Spezialfall:** wenn der Extractor die
Methode `get_markers(pdf_path)` implementiert (nur `pymupdf_structured`
tut das), schreibt die Pipeline die gefundenen Section-Marker in eine
separate Datei. Die Structured-Pipeline liest sie später, um Tabellen und
Figuren mit Abschnitts-Metadaten zu verknüpfen. Mehr dazu in Kapitel 7.

### 6.3 StructuredPipeline

`src/core/pipelines/structured.py`

```
┌──────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐
│ PDF  │───▶│ Structured  │───▶│ Formula     │───▶│ Figure   │
│      │    │ Parser      │    │ Extractor   │    │ Descript.│
│      │    │             │    │ (enrich)    │    │ (enrich) │
│      │    │ pdfplumber/ │    │             │    │          │
│      │    │ MinerU2.5   │    │ pix2tex/    │    │ qwen2.5- │
│      │    │             │    │ ppformulanet│    │ VL/noop  │
└──────┘    └─────────────┘    └─────────────┘    └──────────┘
                   │                   │                │
                   ▼                   ▼                ▼
            (tables, formulas, figures) werden einzeln embedded
                               │
                               ▼
              3 Qdrant Collections: tables, formulas, figures
```

Die Struktur-Pipeline ist die komplexeste, weil sie **drei verschiedene
Element-Typen** verarbeitet. Jeder Typ hat einen eigenen "Anreicherungs"-
Schritt, der optional ist:

- **Tables**: Parser liefert schon Markdown + Rows → nur noch embedden
- **Formulas**: Parser liefert entweder Bounding-Box-only (dann muss der
  `FormulaExtractor` einen Crop machen und LaTeX erkennen) oder direkt
  LaTeX (dann wird der FormulaExtractor übersprungen, weil `formula.latex`
  schon gesetzt ist — siehe `structured.py:138`)
- **Figures**: Parser liefert Bild-Pfade, der `FigureDescriptor`
  (Qwen2.5-VL / moondream / noop) generiert eine Text-Beschreibung, die
  dann eingebettet wird

**Zwei Subtilitäten:**

1. **Figure-Persistenz** (`structured.py:110`): Der Parser schreibt Bilder
   in sein eigenes Tempdir. Die Pipeline verschiebt die Bilder **vor**
   dem Aufruf des VLM in den endgültigen `run_dir/figures/`, sonst wären
   sie weg, bevor der Descriptor sie lesen kann.

2. **Embedding-Fallback**: `Figure` wird auf `description` embedded, wenn
   vorhanden, sonst auf `caption`, sonst auf leeren String
   (`structured.py:181`). `Formula` bevorzugt LaTeX, fällt auf Plaintext
   zurück (`structured.py:170`). Das heißt, dass jedes Element in der
   Vektorsuche findbar ist, egal ob Enrichment funktioniert hat.

---

## 7. Section-Awareness

Das ist die Neuerung aus dem letzten Feature-Branch. Die Idee: Wenn ich
einen Chunk zurückbekomme, will ich wissen "aus welchem Abschnitt stammt
der?", damit ich ihn bei der Traceability einem Kapitel zuordnen kann
(`Claim → Abschnitt → Quelle`).

### Die 4 Bausteine

```
┌─────────────────────┐         ┌────────────────────────┐
│ 1. BaseElement      │         │ 2. utils/sections.py   │
│ section_title       │         │                        │
│ section_path        │         │ SectionMarker          │
│ heading_level       │         │ detect_from_toc()      │
│ (alle Elemente      │         │ detect_from_fonts()    │
│  können das haben)  │         │ assign_sections()      │
└──────────┬──────────┘         │ write/load_sections()  │
           │                    └───────────┬────────────┘
           │                                │
           │                                ▼
┌──────────┴──────────────────────┐  ┌──────────────────────────┐
│ 3. adapters/ocr/                │  │ 4. adapters/chunkers/    │
│    pymupdf_structured.py        │  │    section_aware.py      │
│                                 │  │                          │
│ - get_text("dict") für          │  │ - splittet nie über       │
│   Font-Metadaten                │  │   Section-Grenzen        │
│ - TOC zuerst, Font-Heur.        │  │ - erbt section_title     │
│   als Fallback                  │  │   von Quell-Block        │
│ - setzt sections auf Chunks     │  │                          │
└─────────────────────────────────┘  └──────────────────────────┘
```

### Warum TOC-first + Font-Heuristik als Fallback?

Eine PDF-Datei *kann* eine Outline (Table of Contents) eingebaut haben. Wenn
ja: `doc.get_toc()` liefert die Hierarchie direkt — **das ist Ground Truth,
kein Raten**. Wenn nicht (alte oder schlampig erzeugte PDFs), müssen wir
schätzen. Die Heuristik:

1. Alle Spans sammeln, ihre Schriftgröße → **Median = Body-Text**
2. Alles was ≥ `size_ratio × median` ist (Default 1.3×) ODER Bold-Flag (Bit
   16 in `span.flags`) → Heading-Kandidat
3. Unterschiedliche Heading-Größen werden in bis zu `max_levels` (Default 4)
   Buckets sortiert — größte = Level 1

Das findet in Wissenschafts-PDFs ziemlich zuverlässig die Kapitel- und
Unterkapitel-Überschriften, ohne dass wir einen ML-Klassifizierer brauchen.

### Wie Sections auf Chunks landen

`assign_sections(blocks, markers)` ist in-place:

1. Blöcke werden sortiert nach `(page_number, bbox.y0)` — Lesereihenfolge
2. Marker werden genauso sortiert
3. Mit einem 2-Pointer-Scan läuft man durch beide Listen: jeder Block erbt
   den **zuletzt passierten** Marker
4. Jeder Block erhält `section_title` (der letzte passierte Titel) und
   `section_path` (die komplette Hierarchie, z. B. `["2 Methods",
   "2.3 Data"]`)

Code:

```python
# src/utils/sections.py:127 (vereinfacht)
for block in sorted(blocks, key=lambda b: (b.page_number, b.bbox.y0 if b.bbox else 0.0)):
    while marker_idx < len(ordered):
        m = ordered[marker_idx]
        if (m.page, m.y0) <= (block_page, block_y):
            current = m
            marker_idx += 1
        else:
            break
    if current is not None:
        block.section_title = current.title
        block.section_path = list(current.path)
```

### Der Section-Aware-Chunker

`section_aware.py` macht im Prinzip das, was `fixed_size` auch macht, mit
einem wesentlichen Unterschied:

- Der Chunker sieht beim Splitten eines Blocks auf dessen `section_title`-
  Feld und **propagiert es auf jeden Unter-Chunk**.
- Weil der Extractor bereits jedem rohen Block einen Section-Kontext
  gesetzt hat, schneidet ein Block-Split niemals eine Section-Grenze —
  Block-Grenzen sind schon auf Section-Grenzen ausgerichtet.

### Cross-Pipeline-Linkage via `sections.json`

Die Structured-Pipeline läuft **nach** der Text-Pipeline (siehe
`__main__.py:103-109`). Damit Tables/Figures/Formulas dieselben
Section-Metadaten bekommen wie Chunks, persistiert die Text-Pipeline ihre
Sections in `sections.json`:

```python
# src/core/pipelines/text.py:71 (vereinfacht)
if hasattr(self.extractor, "get_markers"):
    markers = self.extractor.get_markers(pdf_path)
    if markers:
        write_sections(run_dir / "sections.json", markers)
```

Die Structured-Pipeline sucht den **neuesten Text-Run** über
`StorageManager.latest_text_sections(doc_id)`, lädt die Marker und wendet
`assign_sections` auf die eigenen Elemente an:

```python
# src/core/pipelines/structured.py:91 (vereinfacht)
sections_path = self.storage.latest_text_sections(doc_meta.doc_id)
if sections_path is not None:
    markers = load_sections(sections_path)
    all_elements = [*tables, *formulas, *figures]
    assign_sections(all_elements, markers)
    section_source = sections_path.parent.name   # für manifest
```

**Warum über eine JSON-Datei und nicht per Parameter?** Weil die zwei
Pipelines grundsätzlich unabhängig voneinander laufen können sollen (auch
in unterschiedlichen Prozessen). Die Datei ist ein stabiles, lesbares
Hand-Over-Format — und falls sie fehlt, arbeitet die Structured-Pipeline
stumm weiter, nur ohne Section-Felder.

---

## 8. Storage-Layout

`src/utils/storage.py` (`StorageManager`-Klasse) definiert, wie Dateien
auf Disk aussehen:

```
outputs/
└── documents/
    └── <doc_id>/                               # 16-char SHA256 vom Pfad
        ├── document.json                       # Metadaten + Run-Index
        └── runs/
            ├── visual_clip_20260410_143501/
            │   ├── pages/
            │   │   ├── p0000.png
            │   │   ├── p0001.png
            │   │   └── …
            │   ├── elements.jsonl              # VisualPage-Objekte
            │   └── manifest.json               # Run-Metadaten
            │
            ├── text_pymupdf_structured_minilm_20260410_143513/
            │   ├── raw_blocks.jsonl            # Rohe Extractor-Ausgabe
            │   ├── chunks.jsonl                # Gechunkte Endergebnisse
            │   ├── sections.json               # Section-Marker (neu!)
            │   └── manifest.json
            │
            └── structured_pdfplumber_pix2tex_noop_20260410_143527/
                ├── figures/
                │   ├── <doc>_p0_f0.png
                │   └── …
                ├── tables.jsonl
                ├── formulas.jsonl
                ├── figures.jsonl
                └── manifest.json
```

### Warum "Run"-Verzeichnisse und nicht einfach ein Haufen Dateien?

Weil das gleiche PDF mehrmals mit verschiedenen Tool-Kombinationen
ingestiert werden kann, und jede Kombination soll ihre eigenen Ergebnisse
haben. Der Name enthält:

`<pipeline>_<tool_suffix>_<YYYYmmdd_HHMMSS>`

- **pipeline**: `visual` | `text` | `structured`
- **tool_suffix**: kombiniert aus den verwendeten Tools — so kann man auf
  einen Blick sehen, welche Tool-Kombi da drin ist
- **Timestamp**: garantiert Eindeutigkeit über wiederholte Runs

### `document.json` — der Index aller Runs

```json
{
  "doc_id": "a1b2c3d4e5f6g7h8",
  "source_file": "/path/to/test.pdf",
  "runs": [
    {"run_id": "visual_clip_20260410_143501", "pipeline": "visual", "recorded_at": "…"},
    {"run_id": "text_pymupdf_structured_minilm_20260410_143513", "pipeline": "text", "recorded_at": "…"},
    {"run_id": "structured_pdfplumber_pix2tex_noop_20260410_143527", "pipeline": "structured", "recorded_at": "…"}
  ]
}
```

Wird durch `StorageManager.update_document_index` idempotent gepflegt —
wenn ein Run-ID schon drin ist, wird er nicht doppelt eingetragen.

Und wichtig: `latest_text_sections(doc_id)` benutzt genau diesen Index, um
den *neuesten* Text-Run zu finden und zu prüfen, ob sein `sections.json`
existiert.

### `manifest.json` — pro Run

```json
{
  "run_id": "text_pymupdf_structured_minilm_20260410_143513",
  "pipeline": "text",
  "doc_id": "a1b2c3d4e5f6g7h8",
  "started_at": "…",
  "finished_at": "…",
  "duration_seconds": 4.21,
  "tools": {"extractor": "pymupdf_structured", "chunker": "section_aware", "embedder": "minilm"},
  "tool_versions": {"pymupdf_structured": "1.0"},
  "config": {"section_source": "text_pymupdf_structured_minilm_20260410_143513"},
  "counts": {"raw_blocks": 128, "chunks": 204},
  "qdrant": {"collection": "text_chunks", "upserted": 204}
}
```

Das reicht, um nachher beim Benchmarking zwei Runs zu vergleichen (welches
Tool war schneller, welches hat mehr Elemente gefunden).

### JSONL statt ein großes JSON

Die Element-Listen (`chunks.jsonl`, `tables.jsonl`, …) werden als **JSON
Lines** geschrieben: eine Zeile = ein Pydantic-Objekt als JSON. Das ist
bewusst so, damit:

- Große Dokumente mit tausenden Chunks nicht zu Multi-MB-Monster-JSONs
  werden, die man nur komplett laden kann
- Streaming-Lesen möglich ist (`read_jsonl` in `utils/jsonl.py` ist ein
  Generator)
- Man mit `head`, `jq -c`, `grep` gezielt reinschauen kann

---

## 9. Qdrant-Indexing

`src/adapters/vectordb/qdrant.py` enthält **zwei Klassen**, beide
registriert unter `"qdrant"`:

- **`QdrantIndexWriter`** — schreibt Elemente in Collections (implementiert
  `IndexWriter`-Protokoll)
- **`QdrantRetrievalEngine`** — liest Elemente zurück (implementiert
  `RetrievalEngine`-Protokoll)

### Collection-Schema

Pro Element-Typ eine Collection. Jede Collection hat ihre eigene
`VectorParams`:

- `visual_pages`: je nach Embedder Multi-Vektor (MaxSim) oder Single-Vektor
  (Cosine), Dim = `embedder.embedding_dim`
- `text_chunks`, `tables`, `formulas`, `figures`: alle Single-Vector Cosine,
  Dim = `text_embedder.embedding_dim` (also z. B. 384 bei MiniLM)

**Wichtig:** Wenn du das Text-Embedding-Modell wechselst (von MiniLM 384d
auf BGE-M3 1024d), sind **die existierenden Collections nicht mehr gültig**.
Qdrant lässt sich zur Laufzeit nicht umdimensionieren. Der richtige Weg ist,
in der Config einen `collection_prefix` zu setzen (z. B. `dev_bge_`) oder
die Collection zu droppen und neu anzulegen. `ensure_collection` ist
idempotent: wenn die Collection schon existiert, passiert nichts — dort wäre
also der falsche Platz für Migration.

### Payload-Schema (was wird neben dem Vektor gespeichert)

Das ist der **Kern der Rekonstruierbarkeit**. Beim Upsert wird zu jedem
Vektor ein Payload mitgespeichert, das später vollständig ausreicht, um
das Pydantic-Objekt wieder aufzubauen (`_payload_to_element` in `qdrant.py:266`).

```python
# src/adapters/vectordb/qdrant.py:35
def _base_payload(element):
    return {
        "doc_id": element.doc_id,
        "source_file": element.source_file,
        "page_number": element.page_number,
        "object_type": element.object_type,
        "tool_name": element.tool_name,
        "tool_version": element.tool_version,
        "bbox": element.bbox.model_dump() if element.bbox else None,
    }
```

Pro Typ kommen noch spezifische Felder dazu: `content` bei TextChunk/Table,
`latex + content` bei Formula, `image_path + description + caption` bei
Figure. Damit kann der Retriever später (ohne die JSONL-Dateien zu lesen)
vollständige Pydantic-Objekte zurückliefern.

**Aktuell wird `section_title`/`section_path` nicht in den Payload
gelegt** — das ist einer der offenen Nachzügler der Section-Awareness-
Arbeit: die Felder landen zwar im JSONL, aber noch nicht in Qdrant. Wenn du
das später ergänzen willst: ein zusätzliches Feld in `_base_payload` plus
eine Zeile in `_payload_to_element`.

---

## 10. Retrieval & Fusion

Jetzt die andere Richtung: **wie eine Query über alle fünf Collections geht
und die Ergebnisse wieder zusammenfließen**.

### Der UnifiedRetriever

`src/core/retrieval.py` (62 Zeilen) ist erstaunlich kurz:

```python
# src/core/retrieval.py:39
def query(self, query: str, top_k: int = 10, weights=None) -> list[FusionResult]:
    visual_emb = self.visual_embedder.embed_query(query)
    text_emb = self.text_embedder.embed_query(query)

    result_lists = [
        self.retrieval_engine.search_visual(self.visual_collection, visual_emb, top_k),
        self.retrieval_engine.search_text(self.text_collection, text_emb, top_k),
        self.retrieval_engine.search_text(self.tables_collection, text_emb, top_k),
        self.retrieval_engine.search_text(self.formulas_collection, text_emb, top_k),
        self.retrieval_engine.search_text(self.figures_collection, text_emb, top_k),
    ]

    return self.fusion_engine.fuse(result_lists, weights)
```

**Die Query wird zweimal embedded:** einmal vom Visual-Embedder (für die
`visual_pages`-Collection, die andere Vektor-Dimensionen hat) und einmal
vom Text-Embedder (für die vier textbasierten Collections). Dann werden
fünf parallele Qdrant-Queries abgesetzt, und die fünf Result-Listen gehen
in die Fusion.

### Der Trick mit dem Text-Embedder für Tables/Formulas/Figures

Ein kleines Detail, das leicht übersehen wird: **Tables, Formulas und
Figures werden mit demselben Text-Embedder indexiert, mit dem auch die
Text-Chunks indexiert werden**. Deshalb kann eine einzige Text-Query sie
alle finden, solange sie im selben Embedding-Raum leben.

- Tables embedded auf ihrem Markdown-Content
- Formulas embedded auf ihrem LaTeX (Fallback: Plaintext)
- Figures embedded auf ihrer VLM-Beschreibung (Fallback: Caption, leerer
  String)

### Reciprocal Rank Fusion (RRF)

`src/adapters/fusion/rrf.py` (61 Zeilen).

Die Formel ist denkbar einfach:

```
score(element) = Σ  weight_i / (k + rank_in_list_i)
                 i
```

Mit `k = 60` (der Standard-Wert aus dem RRF-Paper, Cormack et al., SIGIR
2009).

**Warum RRF und nicht einfach die Scores summieren?**

Weil die fünf Collections in **vollkommen unterschiedlichen Score-Spaces**
leben:

- Visual (CLIP/ColQwen): Cosine-Similarities auf normalisierten Vektoren,
  typischerweise `[0.15, 0.35]`
- Text (MiniLM/BGE-M3): auch Cosine, typischerweise `[0.4, 0.85]` weil
  MiniLM konzentrierter im hochskaligen Bereich ist
- Formula/Table: ähnlich wie Text, aber mit unterschiedlichen
  Wertebereichen

Wenn du naiv addierst, gewinnt automatisch der Collection mit den
betragsmäßig größten Scores — unabhängig von Relevanz. **RRF eliminiert
diese Verzerrung**, weil nur der *Rang* innerhalb jeder Liste zählt, nicht
der absolute Score-Wert.

Praktisch heißt das: wenn ein Element in einer Liste auf Rang 3 steht,
bekommt es `1/(60+3) ≈ 0.0159`. Ist es in mehreren Listen vorne, addieren
sich die Beiträge — so werden **cross-collection Treffer** stark bevorzugt,
was einen sehr starken Relevanz-Indikator darstellt.

**Die `weights`-Option** erlaubt dir, eine Collection zu bevorzugen, z. B.
`[2.0, 1.0, 1.0, 1.0, 1.0]` würde Visual-Treffer doppelt so hoch wiegen.
Default: gleiche Gewichte.

### Alternative: `score_norm`

`src/adapters/fusion/score_norm.py` ist ein alternatives Fusion-Adapter,
registriert unter `"score_norm"`. Statt nur über den Rang zu arbeiten,
normalisiert es die rohen Scores per Min-Max auf `[0,1]` und rechnet dann
einen gewichteten Durchschnitt. Das **behält relative Score-Magnituden
innerhalb einer Liste** und ist nützlich, wenn deine Retriever-Scores
kalibriert sind und semantisches Gewicht tragen. Für ungewisse Gewichtung
zwischen sehr unterschiedlichen Retrieval-Systemen ist RRF robuster.

Switchen geht per Config:

```yaml
retrieval:
  fusion_engine: score_norm   # statt "rrf"
```

### Der FusionResult-Datentyp

```python
# src/core/models/results.py:17
class FusionResult(BaseModel):
    element: ExtractedElement   # das ursprüngliche Pydantic-Objekt
    fused_score: float          # RRF-Gesamtscore
    source_scores: dict[str, float]  # pro Collection der Original-Score
    rank: int | None = None     # finaler Rang in der gefusten Liste
```

Die `source_scores` sind wichtig für Debugging/Analyse: du siehst direkt,
ob ein Element wegen Visual, Text oder beidem hoch gerankt wurde.

---

## 11. Config-System

`src/core/config.py` ist ein einfaches Pydantic-Schema, das **die komplette
`config.yaml` in getypte Objekte validiert**. Der Top-Level sieht so aus:

```python
# src/core/config.py:56
class AppConfig(BaseModel):
    storage: StorageConfig
    pipelines: PipelinesConfig
    retrieval: RetrievalConfig
    adapters: dict[str, dict[str, Any]]
```

### Zwei-Schicht-Struktur in `pipelines:`

```yaml
pipelines:
  text:
    extractor: "pymupdf_structured"  # adaptername
    chunker: "section_aware"
    embedder: "minilm"
```

Die Pipeline-Config sagt **welcher Adapter** in welcher Rolle verwendet
wird — nur Strings. Die **Konfigurations-Details** für jeden Adapter
kommen separat:

```yaml
adapters:
  pymupdf_structured:
    heading_size_ratio: 1.3
    max_heading_levels: 4
  section_aware:
    chunk_size: 512
    chunk_overlap: 64
  minilm:
    model_name: "all-MiniLM-L6-v2"
    device: "cpu"
    batch_size: 64
```

### Wie der Registry-Lookup an die Adapter-Config kommt

```python
# src/__main__.py:80
get_text_extractor(
    tc.extractor,                               # "pymupdf_structured"
    **get_adapter_config(cfg, tc.extractor),    # {"heading_size_ratio": 1.3, …}
)
```

`get_adapter_config` (in `config.py:78`) ist ein trivialer Wrapper um
`cfg.adapters.get(name, {})` — ein leeres Dict wenn der Adapter keinen
eigenen Config-Block hat. So können Adapter ohne Parameter (`pymupdf_text:
{}`) einfach nicht registriert werden und fallen auf ihre Defaults zurück.

### Was du in der Praxis änderst

**Verschiedene Tool-Kombis ausprobieren** (ohne Code-Änderung):

```yaml
# Variante A: komplett lokal, schnell, AMD/CPU
pipelines:
  visual:
    embedder: "clip"          # statt colqwen25
  text:
    extractor: "pymupdf_text" # statt olmocr2
    embedder: "minilm"        # statt bge_m3
  structured:
    parser: "pdfplumber"      # statt mineru25
    formula_extractor: "pix2tex"
    figure_descriptor: "noop"

# Variante B: High-Quality, CUDA nötig
pipelines:
  visual:
    embedder: "colqwen25"
  text:
    extractor: "olmocr2"
    embedder: "bge_m3"
  structured:
    parser: "mineru25"
    formula_extractor: "ppformulanet"
    figure_descriptor: "qwen25vl"
```

**Qdrant lokal vs. remote:**

```yaml
adapters:
  qdrant:
    host: ":memory:"     # für Tests, alles im RAM
    # host: "localhost"
    # port: 6333
    collection_prefix: "dev_"  # falls du eine gemeinsame DB hast
```

---

## 12. End-to-End-Walkthrough

Das ist der Teil, der dir helfen soll, den Code "in Bewegung" zu verstehen:
ich gehe einen kompletten Lauf von `python -m src ingest test.pdf` durch,
Schritt für Schritt, und zeige bei jedem Schritt das Stück Code, das gerade
aktiv ist.

### Schritt 0: Import-Kaskade

```python
# src/__main__.py:8
import src.adapters
```

Dieser eine Import lädt alle `adapters/*/__init__.py`, die wiederum alle
konkreten Adapter-Module laden. Dadurch werden **alle `@register_*`-
Decorators ausgeführt** und alle Registries sind gefüllt. Danach sind
Lookups über `get_text_extractor("…")` usw. verfügbar.

Ohne diese Zeile wäre der Registry leer.

### Schritt 1: CLI-Parsing + Config-Load

```python
# src/__main__.py:31
args = _parse_args()
cfg = _load_cfg(args.config)        # config.yaml → AppConfig via Pydantic
```

Wenn `--config config.yaml` fehlt, kommen die Pydantic-Defaults zum Zug
(die nehmen aber `olmocr2`, `bge_m3`, `mineru25` an, was auf AMD nicht
läuft — also immer explizit eine Config angeben).

### Schritt 2: Pipelines verdrahten

```python
# src/__main__.py:58
def _run_ingest(pdf_path, cfg):
    storage = StorageManager(cfg.storage.base_dir)   # kennt das outputs/-Layout

    renderer = get_renderer(vc.renderer, **get_adapter_config(cfg, vc.renderer))
    #                       │            │
    #                       │            └── {"dpi": 150}
    #                       └── "pymupdf"
    # ergibt: PyMuPDFRenderer(dpi=150)

    doc_meta = _build_doc_meta(pdf_path, renderer)
    # generate_doc_id(pfad) + renderer.page_count(pfad) + filesize
```

Jeder Lookup zieht **genau die Klasse** aus dem Registry, die im YAML
konfiguriert ist, und übergibt die richtigen kwargs. Wenn `config.yaml`
kein `adapters:`-Block hat, bekommen die Adapter leere Dicts und nehmen
ihre Defaults.

Danach werden drei Pipeline-Objekte gebaut (`VisualPipeline`,
`TextPipeline`, `StructuredPipeline`), jeweils mit allen Adaptern per
Constructor-Injection.

### Schritt 3: VisualPipeline läuft

```python
# src/__main__.py:103
pages = visual_pipeline.run(pdf_path, doc_meta)
```

Jetzt sind wir in `src/core/pipelines/visual.py:44`:

```python
def run(self, pdf_path, doc_meta):
    run_dir = self.storage.run_dir(doc_meta.doc_id, "visual", self.embedder.tool_name)
    #                              │               │         │
    #                              │               │         └── "clip"
    #                              │               └── "visual"
    #                              │
    # erzeugt: outputs/documents/<doc_id>/runs/visual_clip_20260410_143501/
```

Dann wird Collection garantiert (`ensure_collection`), dann alle Seiten
gerendert (`renderer.render_all`), pro Seite embedded (`embedder.embed_page`),
pro Seite als `VisualPage`-Objekt angelegt mit stabilem ID, alles in
`run_dir/pages/` als PNG gespeichert. Am Ende **ein Qdrant upsert-Batch**,
dann `elements.jsonl` und `manifest.json` geschrieben, und das
`document.json` des Dokuments bekommt einen neuen Run-Eintrag.

### Schritt 4: TextPipeline läuft

```python
# src/__main__.py:106
chunks = text_pipeline.run(pdf_path, doc_meta)
```

In `src/core/pipelines/text.py:44`:

```python
raw_blocks = self.extractor.extract_all(pdf_path, doc_meta.doc_id)
# pymupdf_structured öffnet das PDF, baut Section-Marker (TOC oder Fonts),
# zerlegt Seiten in Blöcke mit BBox, annotiert jeden Block mit section_*

write_jsonl(run_dir / "raw_blocks.jsonl", raw_blocks)

# Der Section-Hook: wenn Extractor Marker liefern kann, persistieren
if hasattr(self.extractor, "get_markers"):
    markers = self.extractor.get_markers(pdf_path)
    if markers:
        write_sections(run_dir / "sections.json", markers)

chunks = self.chunker.chunk(raw_blocks)
# section_aware: jeder Block wird gesplittet (wenn > chunk_size),
# Unter-Chunks erben section_title, section_path, heading_level

chunks = self._embed(chunks)        # minilm.embed(texts)
self.index_writer.upsert_text(self.config.collection, chunks)
self._write_outputs(run_dir, raw_blocks, chunks, manifest)
```

Am Ende gibt es im Run-Dir:

```
runs/text_pymupdf_structured_minilm_<ts>/
├── raw_blocks.jsonl
├── chunks.jsonl
├── sections.json           ← neu durch pymupdf_structured
└── manifest.json
```

### Schritt 5: StructuredPipeline läuft

```python
# src/__main__.py:109
tables, formulas, figures = structured_pipeline.run(pdf_path, doc_meta)
```

Hier wird's interessant, weil die Pipeline **nach** der Text-Pipeline läuft
und somit `sections.json` existiert:

```python
# src/core/pipelines/structured.py:83
tables, formulas, figures = self.parser.parse(pdf_path, doc_meta.doc_id)

figures = self._persist_figures(figures, run_dir)

# Section-Linkage aus dem neuesten Text-Run
sections_path = self.storage.latest_text_sections(doc_meta.doc_id)
section_source = None
if sections_path is not None:
    markers = load_sections(sections_path)
    all_elements = [*tables, *formulas, *figures]
    assign_sections(all_elements, markers)
    section_source = sections_path.parent.name   # für manifest

# Formula-Enrichment (wenn nur BBox vorhanden, pix2tex laufen lassen)
formulas = self._enrich_formulas(formulas, pdf_path)
# Figure-Enrichment (VLM-Beschreibung)
figures = self._enrich_figures(figures)

# Embeddings
tables = self._embed_tables(tables)
formulas = self._embed_formulas(formulas)
figures = self._embed_figures(figures)

# Index-Writes in drei separate Collections
self.index_writer.upsert_tables(cols.tables, tables)
self.index_writer.upsert_formulas(cols.formulas, formulas)
self.index_writer.upsert_figures(cols.figures, figures)
```

### Schritt 6: Fertig

Im Terminal steht:

```
Ingesting test.pdf | doc_id=a1b2c3d4e5f6g7h8 | pages=14
  Visual:     14 pages indexed
  Text:       204 chunks indexed
  Structured: 3 tables, 7 formulas, 5 figures indexed

  Outputs: outputs/documents/a1b2c3d4e5f6g7h8
```

Auf der Disk:

```
outputs/documents/a1b2c3d4e5f6g7h8/
├── document.json                # drei Runs drin
└── runs/
    ├── visual_clip_20260410_143501/
    ├── text_pymupdf_structured_minilm_20260410_143513/
    └── structured_pdfplumber_pix2tex_noop_20260410_143527/
```

Und in Qdrant:

- `visual_pages`: 14 Points
- `text_chunks`: 204 Points
- `tables`: 3 Points
- `formulas`: 7 Points
- `figures`: 5 Points

Jeder Point hat eine stabile, deterministische ID — wenn du das PDF noch
mal ingestierst, werden die Points **überschrieben, nicht dupliziert**.

### Schritt 7: Eine Query (nicht im CLI, aber so würde's aussehen)

```python
from src.core.retrieval import UnifiedRetriever
from src.core.registry import get_retrieval_engine, get_fusion_engine, \
    get_text_embedder, get_visual_embedder

retriever = UnifiedRetriever(
    retrieval_engine=get_retrieval_engine("qdrant", host=":memory:"),
    visual_embedder=get_visual_embedder("clip", device="cpu"),
    text_embedder=get_text_embedder("minilm", device="cpu"),
    fusion_engine=get_fusion_engine("rrf"),
    visual_collection="visual_pages",
    text_collection="text_chunks",
    tables_collection="tables",
    formulas_collection="formulas",
    figures_collection="figures",
)

results = retriever.query("energy balance equation for solar panels", top_k=10)
for r in results:
    print(r.rank, r.fused_score, r.element.object_type, r.element.page_number)
    print("  sources:", r.source_scores)
```

Ausgabe könnte so aussehen:

```
1  0.0328  formula     7   sources: {'formulas': 0.82, 'text_chunks': 0.74}
2  0.0294  text_chunk  6   sources: {'text_chunks': 0.79}
3  0.0258  figure      8   sources: {'figures': 0.65, 'visual_pages': 0.41}
...
```

Die Formel auf Seite 7 ist ganz oben, weil sie **sowohl im `formulas`-
als auch im `text_chunks`-Retrieval** auftauchte → zwei RRF-Beiträge
summieren sich, cross-collection Bonus.

---

## 13. Tests

`tests/` enthält 16 Testdateien, 140 Tests insgesamt. Die Struktur spiegelt
die Architektur:

| Testdatei | Was wird getestet |
|---|---|
| `test_models.py` | Pydantic-Modelle, BoundingBox-Math, discriminated union |
| `test_ids.py` | ID-Determinismus, Kollisions-Verhalten |
| `test_jsonl.py` | JSONL-Writer/Reader Roundtrip |
| `test_manifest.py` | ManifestBuilder korrekt serialisiert |
| `test_storage_layout.py` | StorageManager erzeugt Verzeichnisse, `document.json` wird korrekt aktualisiert |
| `test_config.py` | YAML → AppConfig → Defaults |
| `test_chunker.py` | FixedSizeChunker (fixed_size.py) |
| `test_section_aware_chunker.py` | SectionAwareChunker, Section-Vererbung |
| `test_sections.py` | SectionMarker, TOC+Font-Detection, assign_sections, write/load |
| `test_pymupdf_structured.py` | PyMuPDFStructuredExtractor mit gemocktem fitz |
| `test_amd_adapters.py` | CPU-fähige Adapter (clip, minilm, pdfplumber, pix2tex, moondream) |
| `test_pipelines.py` | End-to-End-Pipelines gegen In-Memory-Qdrant |
| `test_rrf.py` | RRF-Fusion, Rangierung, Gewichte |
| `test_score_norm.py` | Min-Max-Fusion |
| `test_retrieval.py` | UnifiedRetriever ruft alle 5 Collections ab und fused |

### Wie die Pipelines ohne echte ML-Modelle getestet werden

`test_pipelines.py` benutzt **In-Memory-Qdrant** (`host=":memory:"`)
zusammen mit den **AMD/CPU-Adaptern**, die real sind aber klein und schnell:

- `pymupdf_text` → reine Python, keine ML
- `pdfplumber` → reine Python
- `minilm` → 384-dim CPU-Embedding, ~80MB, lädt in Sekunden
- `clip` → 512-dim CPU-Vision, ~150MB
- `pix2tex` → CPU ViT für LaTeX
- `moondream` / `noop` → kleiner VLM oder Dummy

Das heißt, die Tests laufen **ohne CUDA, ohne olmOCR2, ohne ColQwen2.5**
und bilden trotzdem den kompletten Pipeline-Fluss ab. Zeit: ca. 3 Sekunden
für 140 Tests.

---

## Anhang: die wichtigsten Dateien im Überblick

| Pfad | Zeilen | Zweck |
|---|---|---|
| `src/core/registry.py` | 78 | Herz der Modularität — Register + Lookup-Factories |
| `src/core/config.py` | 85 | Pydantic-Schema für YAML |
| `src/core/models/elements.py` | 95 | BaseElement + 5 Subklassen + Discriminated Union |
| `src/core/pipelines/text.py` | ~115 | TextPipeline |
| `src/core/pipelines/visual.py` | 116 | VisualPipeline |
| `src/core/pipelines/structured.py` | ~215 | StructuredPipeline |
| `src/core/retrieval.py` | 62 | UnifiedRetriever |
| `src/adapters/vectordb/qdrant.py` | ~300 | Writer + Retriever + Payload-Schema |
| `src/adapters/fusion/rrf.py` | 61 | Reciprocal Rank Fusion |
| `src/utils/storage.py` | ~115 | Run-Dir-Layout + document.json |
| `src/utils/sections.py` | ~180 | SectionMarker, TOC/Font-Detection, assign |
| `src/utils/ids.py` | 22 | SHA256-basierte Element- und Doc-IDs |

**Das solltest du für eine komplette Mental-Map alle einmal gelesen haben.**
Danach ist jede Erweiterung nur noch eine Frage: "Welche Schicht (utils,
models, interfaces, pipelines, adapters)?" → "In welche Datei kommt der
neue Code?"
