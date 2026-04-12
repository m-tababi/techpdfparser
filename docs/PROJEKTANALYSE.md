# Projektanalyse `techpdfparser`

Stand: 2026-04-12

## Ziel dieser Datei

Diese Datei ist keine reine Fehlerliste mehr. Sie soll den **aktuellen technischen Stand** des Projekts so erklären, dass du:

1. verstehst, **wie das Projekt heute aufgebaut ist**
2. nachvollziehen kannst, **welche Architekturprobleme bereits gelöst wurden**
3. weißt, **welche Dateien wofür zuständig sind**
4. das Projekt bei Bedarf **sauber neu aufsetzen** kannst
5. selbst gezielt **Modelle, Backends oder Konfigurationen anpassen** kannst

Die Datei beschreibt also den Stand **nach der Umsetzung** der Architektur-Härtung für modulare Embeddings und modulare Vector-DB-Backends.

## Kurzfassung

Das Projekt ist heute deutlich robuster als im ursprünglichen Stand der Analyse.

Die wichtigsten Verbesserungen sind:

- Es gibt jetzt ein klares **Index-Konzept** mit `VectorSchema`, `adapter_signature` und `index_namespace`.
- Modellwechsel und Backendwechsel sind nicht mehr nur „per Config gedacht“, sondern im Code wirklich abgesichert.
- Bestehende Collections werden jetzt gegen ihr erwartetes Schema geprüft.
- Es gibt mit `doctor` ein Preflight-Kommando, das Setup, Imports und Backend-Gesundheit prüft.
- `section_aware` ist jetzt wirklich section-basiert und nicht nur dem Namen nach.
- Die Pipeline-Manifeste dokumentieren jetzt deutlich besser, **was** gelaufen ist und **wohin** indexiert wurde.
- Das Projekt hat ein testbares Backend-Abstraktionsmodell: produktiv `qdrant`, test-only `memory`.

Wichtig ist aber die Trennung zwischen:

- **Code-Stand im Repository**
- **Zustand des aktuell vorhandenen `venv`**

Im Repository ist die Architektur deutlich verbessert. Das aktuell vorhandene `venv` ist aber weiterhin nicht sauber, solange es nicht mit den neuen Constraints neu aufgebaut wird.

## Was das Projekt aktuell macht

`techpdfparser` verarbeitet technische PDFs über drei getrennte Pipelines:

1. **Visual Pipeline**
   - rendert Seiten als Bilder
   - erzeugt visuelle Embeddings
   - schreibt sie in die visuelle Collection

2. **Text Pipeline**
   - extrahiert Text
   - teilt ihn in Chunks
   - erzeugt Text-Embeddings
   - schreibt sie in die Text-Collection

3. **Structured Pipeline**
   - findet Tabellen, Formeln und Figures
   - reichert sie an
   - embeddet sie
   - schreibt sie in drei eigene Collections

Beim Retrieval werden dann die fünf Sammlungen gemeinsam abgefragt:

- `visual_pages`
- `text_chunks`
- `tables`
- `formulas`
- `figures`

## Die wichtigsten Dateien im aktuellen Stand

Wenn du das System verstehen oder anpassen willst, sind diese Dateien die wichtigsten Einstiegspunkte:

- [src/__main__.py](../src/__main__.py)
  - CLI-Einstiegspunkt
  - `ingest`
  - `doctor`
  - Runtime-Build aus der Config

- [src/core/config.py](../src/core/config.py)
  - Pydantic-Config
  - Defaultwerte
  - Cross-Checks wie `section_aware` nur mit `pymupdf_structured`

- [src/core/indexing.py](../src/core/indexing.py)
  - `VectorSchema`
  - `adapter_signature`
  - Namespace-Auflösung
  - aufgelöstes Index-Layout

- [src/core/interfaces/indexer.py](../src/core/interfaces/indexer.py)
  - Vertrag für Vector-DB-Backends

- [src/adapters/vectordb/qdrant.py](../src/adapters/vectordb/qdrant.py)
  - produktiver Backend-Adapter

- [src/adapters/vectordb/memory.py](../src/adapters/vectordb/memory.py)
  - test-only Backend

- [src/core/pipelines/visual.py](../src/core/pipelines/visual.py)
- [src/core/pipelines/text.py](../src/core/pipelines/text.py)
- [src/core/pipelines/structured.py](../src/core/pipelines/structured.py)
  - die eigentlichen Orchestrierer

- [src/adapters/chunkers/section_aware.py](../src/adapters/chunkers/section_aware.py)
  - section-basiertes Chunking

- [src/utils/storage.py](../src/utils/storage.py)
  - Dateisystem-Layout
  - `document.json`

- [src/utils/manifest.py](../src/utils/manifest.py)
  - Manifeste pro Run

- [config.example.yaml](../config.example.yaml)
  - wichtigste Datei für spätere Anpassungen

## Die zentralen Konzepte einfach erklärt

## 1. Adapter statt harter Tool-Kopplung

Das Projekt ist so gebaut, dass konkrete Tools nicht direkt in die Pipelines eingebrannt sind.

Beispiel:

- Die Text-Pipeline weiß nicht „ich muss immer BGE-M3 benutzen“
- Sie bekommt einfach irgendeinen `TextEmbedder`

Das ist der Kern der Modularität:

- `core/` beschreibt **was** ein Baustein können muss
- `adapters/` liefern **wie** es konkret umgesetzt wird

Wenn du später ein anderes Embedding-Modell oder ein anderes VDB-Backend willst, ist die Idee:

- neuen Adapter bauen
- registrieren
- in der Config auswählen

## 2. `VectorSchema`

Das wichtigste neue Architekturkonzept ist `VectorSchema` aus [src/core/indexing.py](../src/core/indexing.py).

Es beschreibt, wie eine Collection aus Sicht des Projekts aussehen muss:

```python
VectorSchema(
    dim=1024,
    distance="cosine",
    multi_vector=False,
)
```

Bedeutung:

- `dim`: Vektordimension
- `distance`: Distanzmetrik
- `multi_vector`: ob pro Dokument ein Vektor oder mehrere Vektoren gespeichert werden

Warum das wichtig ist:

Vorher konnte eine bestehende Collection einfach weiterverwendet werden, auch wenn das neue Modell gar nicht zum alten Schema passte. Jetzt gibt es einen expliziten Vertrag.

## 3. `adapter_signature`

Jeder Embedder bekommt jetzt eine deterministische Signatur.

Sie setzt sich zusammen aus:

- Tool-Name
- Modellname
- Dimension
- Distanzmetrik
- Single-/Multi-Vector-Modus

Warum das wichtig ist:

- die Signatur landet im Namespace
- die Signatur landet im Manifest
- die Signatur hilft beim Debuggen

Wenn du z. B. von `bge_m3` auf `minilm` wechselst, ändert sich die Signatur.

## 4. `index_namespace`

Collections werden nicht mehr nur über ihren Basenamen angesprochen, sondern über ein **aufgelöstes Layout**.

Beispiel:

- Basisname: `text_chunks`
- Namespace: `qdrant__colqwen...__bge...`
- aufgelöster Name:
  - `qdrant__colqwen...__bge...__text_chunks`

Das wird in [src/core/indexing.py](../src/core/indexing.py) berechnet.

Es gibt drei sinnvolle Modi:

- `auto`
  - sicherster Standard
  - neuer Namespace pro Backend-/Embedder-Kombination

- `legacy`
  - alte Collection-Namen ohne Präfix
  - für Migration oder Rückwärtskompatibilität

- eigener String
  - z. B. `projekt-a`
  - gut, wenn du einen bewusst festen Index-Namen willst

## 5. `doctor`

`doctor` ist das neue Prüfkommando in [src/__main__.py](../src/__main__.py).

Aufruf:

```bash
python -m src doctor --config config.yaml
```

Es prüft:

- ob die Config gebaut werden kann
- ob die ausgewählten Adapter importierbar sind
- ob die Runtime gebaut werden kann
- ob das Backend erreichbar ist
- ob bestehende Collections zum erwarteten Schema passen

Kurz gesagt:

- `doctor` prüft den Gesundheitszustand
- `ingest` macht echte Arbeit

## 6. `document.json` und `manifest.json`

Es gibt zwei Ebenen von Laufzeit-Metadaten:

### `manifest.json`

Pro Pipeline-Run gibt es ein Manifest mit:

- verwendeten Tools
- Tool-Versionen
- Counts
- Backend
- Namespace
- Collections
- Adapter-Signaturen
- Vektor-Schemata

Das ist in [src/utils/manifest.py](../src/utils/manifest.py) umgesetzt.

### `document.json`

Pro Dokument gibt es zusätzlich einen Überblick in [src/utils/storage.py](../src/utils/storage.py):

- welche Runs zu diesem Dokument gehören
- welche Pipeline gelaufen ist
- wann der Run eingetragen wurde
- welches Layout aktiv war

Das ist wichtig für Nachvollziehbarkeit.

## Wie der aktuelle Lauf technisch funktioniert

## 1. Start über CLI

Alles beginnt in [src/__main__.py](../src/__main__.py).

Es gibt aktuell zwei Kommandos:

- `python -m src ingest ...`
- `python -m src doctor ...`

Für `ingest` passiert grob:

1. Config laden
2. Adapter über Registry bauen
3. Index-Layout auflösen
4. Pipelines instanziieren
5. Pipelines nacheinander ausführen

## 2. Runtime-Build

Die Funktion `_build_runtime()` in [src/__main__.py](../src/__main__.py) baut aus der Config die aktive Laufzeit zusammen.

Dabei werden u. a. instanziiert:

- Renderer
- Visual Embedder
- Text Extractor
- Text Chunker
- Text Embedder
- Structured Parser
- Formula Extractor
- Figure Descriptor
- Index Writer
- Retrieval Engine
- Fusion Engine

Und ganz wichtig:

- `resolve_index_layout(...)`

Dadurch kennt die Runtime schon vor dem Ingest:

- welches Backend aktiv ist
- welche Signaturen aktiv sind
- wie die Collection-Namen wirklich heißen
- welche Schemas erwartet werden

## 3. Visual Pipeline

Die Visual Pipeline in [src/core/pipelines/visual.py](../src/core/pipelines/visual.py) macht:

1. Collection-Namen und Schema bestimmen
2. `ensure_collection(...)`
3. Seiten rendern
4. Bilder speichern
5. pro Seite Embedding erzeugen
6. `upsert_visual(...)`
7. Manifest und `document.json` schreiben

Wichtig:

- die Pipeline arbeitet komplett über Interfaces
- die konkrete VDB ist ihr egal
- das konkrete Visual-Modell ist ihr egal

## 4. Text Pipeline

Die Text Pipeline in [src/core/pipelines/text.py](../src/core/pipelines/text.py) macht:

1. Collection und Schema bestimmen
2. `ensure_collection(...)`
3. Rohblöcke extrahieren
4. Rohblöcke sofort als JSONL persistieren
5. falls vorhanden: Section-Marker schreiben
6. Chunking
7. Embeddings erzeugen
8. `upsert_text(...)`
9. Manifest und `document.json` schreiben

Wichtige Robustheitsverbesserung:

- die Anzahl der Embeddings wird hart gegen die Anzahl der Chunks geprüft

Wenn ein Embedder falsch arbeitet, wird nicht mehr still weitergeschrieben.

## 5. Structured Pipeline

Die Structured Pipeline in [src/core/pipelines/structured.py](../src/core/pipelines/structured.py) macht:

1. drei Collections bestimmen: `tables`, `formulas`, `figures`
2. für alle drei `ensure_collection(...)`
3. Parser laufen lassen
4. Figure-Bilder in den Run-Ordner verschieben
5. Section-Informationen aus dem letzten Text-Run übernehmen
6. Formeln und Figures anreichern
7. Tabellen/Formeln/Figures embedden
8. getrennt indexieren
9. Manifest und `document.json` schreiben

Wichtige Details:

- Tabellen, Formeln und Figures nutzen denselben Text-Embedder
- Formeln verwenden bevorzugt LaTeX für die Embeddings
- Figures verwenden bevorzugt Beschreibung oder Caption
- Figure-Dateien werden mit Context Manager geöffnet

## 6. Retrieval

Das Retrieval ist aktuell in [src/core/retrieval.py](../src/core/retrieval.py).

Es ist aktuell bewusst:

- **sequentiell**
- **einfach**
- **robust**

Das heißt:

1. Visual-Query
2. Text-Query
3. Tables-Query
4. Formulas-Query
5. Figures-Query
6. Fusion

Das ist aktuell eine bewusste Designentscheidung und kein Versehen mehr.

## Wie ich die kritischen Punkte aus der ursprünglichen Analyse umgesetzt habe

## 1. Problem: Modellwechsel konnte inkompatible Collections wiederverwenden

Vorher:

- existierende Collection = einfach weiterverwenden

Jetzt:

- `VectorSchema`
- `get_collection_schema()`
- `ensure_collection(..., schema, fail_on_schema_mismatch=...)`

Umgesetzt in:

- [src/core/interfaces/indexer.py](../src/core/interfaces/indexer.py)
- [src/adapters/vectordb/qdrant.py](../src/adapters/vectordb/qdrant.py)
- [src/adapters/vectordb/memory.py](../src/adapters/vectordb/memory.py)

Effekt:

- falsche Dimension oder falscher Multi-Vector-Modus fällt jetzt auf

## 2. Problem: `section_aware` war inhaltlich nicht wirklich section-aware

Vorher:

- Blöcke wurden eher blockweise gesplittet
- Section-Grenzen waren nicht wirklich der zentrale Steuermechanismus

Jetzt:

- Gruppierung nach `(page_number, section_path)`
- benachbarte Blöcke derselben Sektion werden zusammengeführt
- zu große Einzelblöcke werden trotzdem gesplittet
- Metadaten wie `child_ids` und `parent_id` bleiben nachvollziehbar

Umgesetzt in:

- [src/adapters/chunkers/section_aware.py](../src/adapters/chunkers/section_aware.py)

## 3. Problem: Das Projekt war „modular gedacht“, aber das Index-Layout war nicht modular genug

Vorher:

- Collection-Namen waren faktisch zu statisch
- Backend und Embedder wirkten nicht stark genug auf den Index-Namen

Jetzt:

- `ResolvedIndexLayout`
- `apply_namespace(...)`
- `resolve_namespace(...)`
- `layout_metadata(...)`

Effekt:

- das aktive Layout ist explizit
- du kannst nachvollziehen, warum ein Ingest in genau diese Collections schreibt

## 4. Problem: Fehlersuche bei Setup-Problemen war zu schwer

Vorher:

- viele Probleme tauchten erst mitten im Ingest auf

Jetzt:

- `doctor`
- Dependency-Probe
- Runtime-Build-Prüfung
- Backend-Healthcheck
- Schema-Validierung

Effekt:

- viele Fehlkonfigurationen lassen sich vor dem eigentlichen Lauf finden

## 5. Problem: Das Projekt war schlecht nachvollziehbar, wenn man später Runs vergleicht

Vorher:

- zu wenig Metadaten über den aktiven Index-Zustand

Jetzt:

- Manifest enthält Backend, Namespace, Signaturen, Schema
- `document.json` enthält Layout-Metadaten pro Run

Effekt:

- man kann besser nachvollziehen, warum ein bestimmter Run bestimmte Ergebnisse produziert hat

## Wie du das Projekt heute neu aufbauen würdest

Wichtig:

- **kein Docker**
- wenn du das Projekt sauber neu starten willst, bau lieber ein frisches `venv`
- das aktuell vorhandene `venv` ist wegen alter Paketstände nicht verlässlich

Ein sinnvoller Ablauf wäre:

1. neues virtuelles Environment anlegen
2. Requirements installieren, die jetzt `constraints-common.txt` mitziehen
3. `config.example.yaml` nach `config.yaml` kopieren
4. gewünschte Adapter wählen
5. `doctor` laufen lassen
6. erst danach `ingest`

### Praktischer Neuaufbau ohne Docker

Wenn du lokal erst einmal nur einen **sauberen Funktionscheck** willst, ist der einfachste Weg:

1. frisches `venv`
2. CPU-Setup
3. `memory` als Backend
4. `doctor`
5. kleiner Test-Ingest

Beispielablauf:

```bash
python3 -m venv venv-clean
source venv-clean/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt -r requirements-amd.txt
pip install -e .
cp config.example.yaml config.yaml
```

Wichtig:

- diesen Ablauf wirklich in einem **frischen** Environment machen
- nicht versuchen, das alte `./venv` nur teilweise „nachzupatchen“

Danach in `config.yaml` für den einfachsten lokalen Start:

```yaml
retrieval:
  retrieval_engine: "memory"

pipelines:
  visual:
    embedder: "clip"
  text:
    extractor: "pymupdf_structured"
    chunker: "section_aware"
    embedder: "minilm"
  structured:
    parser: "pdfplumber"
    formula_extractor: "pix2tex"
    figure_descriptor: "moondream"
```

Dann:

```bash
python -m src doctor --config config.yaml
python -m src ingest path/to/document.pdf --config config.yaml
```

Warum diese Variante sinnvoll ist:

- kein externer Service nötig
- keine produktive Vector-DB nötig
- du testest trotzdem die komplette modulare Architektur
- du bekommst direkt Dateien, Manifeste und `document.json`

Wenn du später ein persistentes Backend willst, änderst du primär:

- `retrieval.retrieval_engine`
- den passenden Block in `adapters:`

Die Pipelines selbst bleiben gleich.

### Konfigurationsdatei vorbereiten

Ausgangspunkt:

- [config.example.yaml](../config.example.yaml)

Wichtige Felder:

```yaml
retrieval:
  retrieval_engine: "qdrant"
  fusion_engine: "rrf"
  index_namespace: "auto"
  validate_on_start: true
  fail_on_schema_mismatch: true
```

## Wie die erzeugten Dateien auf der Platte aussehen

Das Dateisystem-Layout wird in [src/utils/storage.py](../src/utils/storage.py) verwaltet.

Pro Dokument entsteht grob:

```text
outputs/
  documents/
    <doc_id>/
      document.json
      runs/
        visual_<...>/
          manifest.json
          elements.jsonl
          pages/
        text_<...>/
          manifest.json
          raw_blocks.jsonl
          chunks.jsonl
          sections.json
        structured_<...>/
          manifest.json
          tables.jsonl
          formulas.jsonl
          figures.jsonl
          figures/
```

Das ist wichtig für dich, wenn du später selbst prüfen willst:

- welcher Run wann entstanden ist
- welche Pipeline welche Artefakte erzeugt hat
- welche Sections für den Structured-Run übernommen wurden
- in welchen Namespace indexiert wurde

## Was du selbst leicht anpassen kannst

## 1. Backend wechseln

In der Config:

```yaml
retrieval:
  retrieval_engine: "qdrant"
```

Aktuell sinnvoll:

- `"qdrant"` für produktiv
- `"memory"` für Tests oder lokale Verifikation

Wenn du `"memory"` nutzt:

- kein externer Server nötig
- gleiche Kernlogik bleibt erhalten
- nur Persistenz fehlt

## 2. Text-Embedder wechseln

In der Config:

```yaml
pipelines:
  text:
    embedder: "bge_m3"
```

Alternative:

- `"minilm"`

Wichtig:

- Structured nutzt denselben Text-Embedder
- also ändern sich damit auch die Schemata für
  - `tables`
  - `formulas`
  - `figures`

## 3. Visual-Embedder wechseln

In der Config:

```yaml
pipelines:
  visual:
    embedder: "colqwen25"
```

Alternative:

- `"clip"`

Wichtiger Unterschied:

- `colqwen25` = Multi-Vector
- `clip` = Single-Vector

Genau deshalb ist die Schema-Prüfung jetzt so wichtig.

## 4. Namespace-Strategie anpassen

In der Config:

```yaml
retrieval:
  index_namespace: "auto"
```

Empfehlung:

- `auto` für sicheren Standard
- `legacy` nur wenn du bewusst alte Collections weiterverwenden willst
- eigener String nur wenn du gezielt feste Namen willst

## 5. Section-Aware aktivieren

In der Config:

```yaml
pipelines:
  text:
    extractor: "pymupdf_structured"
    chunker: "section_aware"
```

Wichtig:

- genau diese Kombination ist nötig
- andere Kombinationen werden von der Config-Validierung abgelehnt

## 6. Schema-Strenge ändern

In der Config:

```yaml
retrieval:
  fail_on_schema_mismatch: true
```

Das sollte für robuste Nutzung auf `true` bleiben.

Wenn du experimentell arbeitest, könntest du es temporär lockern. Für das normale Projektverhalten würde ich es nicht abschalten.

## Wenn du später eine andere Vector-DB willst

Die Architektur ist jetzt so vorbereitet, dass du ein neues Backend ergänzen kannst, ohne die Pipelines neu zu schreiben.

Wenn du später z. B. `LanceDB` willst, müsstest du vor allem diese Stellen anfassen:

1. neue Datei in `src/adapters/vectordb/`
2. dort `IndexWriter` und `RetrievalEngine` implementieren
3. in `src/adapters/vectordb/__init__.py` registrieren
4. `config.example.yaml` um `adapters.lancedb` ergänzen
5. in [src/__main__.py](../src/__main__.py) `_DEPENDENCY_MODULES` ergänzen
6. Tests ergänzen

Die Pipelines selbst müssten dafür kaum geändert werden.

Das ist ein gutes Zeichen: Die Modularität ist heute deutlich echter als am Anfang.

## Was noch offen ist

## 1. Das aktuelle `venv` ist noch kaputt

Direkte Imports schlagen im aktuellen vorhandenen `venv` weiter fehl:

- `pdfplumber`
- `qdrant_client`

Das heißt:

- der Repository-Code ist vorbereitet
- das bestehende Environment ist noch nicht aufgeräumt

Das ist aktuell der wichtigste praktische Restpunkt.

## 2. `PyMuPDFRenderer.render_all()` ist noch nicht elegant

In [src/adapters/renderers/pymupdf.py](../src/adapters/renderers/pymupdf.py) wird das PDF für `render_all()` weiterhin mehrfach geöffnet.

Das ist nicht kritisch für die Architektur, aber ein sinnvoller nächster Refactor.

## 3. Qdrant ist weiter das einzige produktive Backend

`memory` ist sehr nützlich, aber kein persistentes Produktivsystem.

Wenn du die Modularität weiter testen willst, wäre ein zweites echtes persistentes Backend der nächste Schritt.

## 4. Retrieval ist absichtlich noch nicht parallelisiert

Das ist heute kein Widerspruch zur Doku mehr, aber natürlich ein möglicher späterer Performance-Hebel.

## 5. Lokale Scratch-Dateien

Die früher vorhandenen `* 2.py`-Duplikate wurden geprüft und entfernt.

Ergebnis:

- vier Dateien waren byte-identische Kopien
- zwei Dateien waren ältere Vorversionen des aktuellen `section_aware`-Stands

Der Tree ist damit in diesem Punkt wieder sauberer, und die Tool-Konfiguration braucht dafür keine Sonderbehandlung mehr.

## Verifizierter Stand

Lokal geprüft:

- `./venv/bin/python -m pytest -q`
  - Ergebnis: `157 passed, 5 warnings`

- `./venv/bin/python -m ruff check src`
  - Ergebnis: sauber

- `./venv/bin/python -m mypy`
  - Ergebnis: `Success: no issues found in 41 source files`

Zusätzlich geprüft:

- direkter `pdfplumber`-Import im aktuellen `venv` schlägt fehl
- direkter `qdrant_client`-Import im aktuellen `venv` schlägt fehl

Das bestätigt die Trennung zwischen:

- **verbessertem Repository-Stand**
- **noch altem lokalen Environment**

## Bottom line

Der aktuelle Stand ist nicht mehr nur „eine gute Idee mit ein paar Adaptern“, sondern eine deutlich robustere, nachvollziehbare und anpassbare Basis.

Das Projekt ist heute so aufgebaut, dass du:

- Embedder per Config wechseln kannst
- Backend per Adapter-Modell wechseln kannst
- Index-Namen bewusst steuern kannst
- Schemafehler früher erkennst
- Runs nachvollziehbarer dokumentiert bekommst

Wenn du das Projekt weiterentwickeln willst, sind die sinnvollsten nächsten Schritte:

1. frisches `venv` mit den neuen Constraints aufsetzen
2. `doctor` als festen Schritt vor jedem Ingest nutzen
3. bewusst entscheiden, ob du `auto`, `legacy` oder eigene Namespaces willst
4. bei Bedarf später ein zweites echtes persistentes Backend ergänzen

Wenn du diese Datei als Arbeitsgrundlage liest, solltest du jetzt nicht nur verstehen **was** geändert wurde, sondern auch **wo** du selbst später eingreifen würdest.
