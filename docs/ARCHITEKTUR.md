# Ausführliche Architektur

Diese Datei beschreibt den aktuellen technischen Stand des Projekts nach der Härtung für modulare Embeddings und modulare Vector-DB-Backends. Sie ersetzt ältere Annahmen wie statische Collection-Namen, implizite Wiederverwendung bestehender Indizes oder paralleles Retrieval.

## 1. Systemziel

Das Projekt extrahiert technische PDFs über drei getrennte Sichten:

- visuell
- textuell
- strukturiert

Das Ergebnis sind indexierte Elemente in einer Vector-DB, die später gemeinsam durchsucht und fusioniert werden können. Die Architektur ist so gebaut, dass konkrete Tools per Konfiguration ausgetauscht werden können, ohne den Pipeline-Code anzupassen.

## 2. Schichtenmodell

```text
utils -> core/models -> core/interfaces -> core/pipelines -> adapters
```

Praktisch bedeutet das:

- `core/interfaces` beschreibt nur Verträge.
- `core/pipelines` arbeitet nur gegen diese Verträge.
- `adapters` liefern die konkreten Implementierungen.
- `src.adapters` wird beim CLI-Start importiert, damit alle Registry-Dekoratoren aktiv sind.

## 3. Die drei Pipelines

### Visual

- rendert PDF-Seiten
- erzeugt pro Seite visuelle Embeddings
- schreibt in die Visual-Collection

Bei Multi-Vector-Modellen wie `colqwen25` ist das Collection-Schema explizit als Multi-Vector markiert.

### Text

- extrahiert Text
- chunkt den Text
- erzeugt Text-Embeddings
- schreibt in die Text-Collection

`section_aware` ist nicht mehr nur ein Name, sondern ein echter Merge-/Split-Algorithmus entlang von `section_path`.

### Structured

- findet Tabellen, Formeln und Abbildungen
- reichert Formeln und Figures an
- nutzt den Text-Embedder für Tabellen, Formeln und Figure-Beschreibungen
- schreibt in drei getrennte Collections

## 4. Registry und Konfiguration

Die Runtime wird vollständig über die Config zusammengesetzt. Namen wie `colqwen25`, `bge_m3`, `qdrant` oder `memory` werden im Registry auf konkrete Klassen aufgelöst.

Wichtige Config-Bereiche:

```yaml
pipelines:
  visual:
    renderer: "pymupdf"
    embedder: "colqwen25"
  text:
    extractor: "pymupdf_structured"
    chunker: "section_aware"
    embedder: "bge_m3"
  structured:
    parser: "mineru25"

retrieval:
  retrieval_engine: "qdrant"
  fusion_engine: "rrf"
  index_namespace: "auto"
  validate_on_start: true
  fail_on_schema_mismatch: true
```

Zusätzliche Validierung:

- `section_aware` ist nur mit `pymupdf_structured` gültig
- falsche Chunking-Konfigurationen schlagen früh fehl

## 5. `VectorSchema` als gemeinsamer Vertrag

Der zentrale Baustein für Modularität ist `VectorSchema` aus [src/core/indexing.py](../src/core/indexing.py):

```python
VectorSchema(dim=1024, distance="cosine", multi_vector=False)
```

Dieses Schema beschreibt backend-agnostisch, was eine Collection erwartet. Der Qdrant-Adapter übersetzt dieses Schema in Qdrants eigenes Modell. Das Memory-Backend nutzt dasselbe Schema direkt.

## 6. Deterministische Adapter-Signaturen

Embedding-Adapter liefern jetzt eine `adapter_signature`, die aus folgenden Teilen entsteht:

- Tool-Name
- Modellname
- Vektordimension
- Distanzmetrik
- Single-/Multi-Vector-Modus

Diese Signatur wird für drei Dinge verwendet:

1. Namespace-Auflösung
2. Manifeste und `document.json`
3. Fehlermeldungen und Diagnose

## 7. Namespaces und Collection-Auflösung

Die Collection-Namen werden nicht mehr direkt aus der Config übernommen, sondern über ein aufgelöstes Layout:

```text
resolved_collection = namespace + "__" + base_collection
```

Namespace-Modi:

- `auto`: aus Backend + Embedder-Signaturen berechnet
- `legacy`: kein Präfix, alte Namen bleiben erhalten
- eigener String: expliziter Namespace

Konsequenz:

- Wechselst du Text-Embedder, Visual-Embedder oder Backend, entsteht im `auto`-Modus automatisch ein neuer Namespace.
- Bestehende Daten bleiben erhalten und werden nicht überschrieben.
- Reindex bedeutet: Config ändern, `doctor` ausführen, PDFs neu ingestieren.

## 8. Backend-Verhalten

### Qdrant

Der produktive Pfad ist [src/adapters/vectordb/qdrant.py](../src/adapters/vectordb/qdrant.py).

Wesentliche Robustheitsregeln:

- `healthcheck()` prüft die Erreichbarkeit des Backends
- `get_collection_schema()` liest das vorhandene Schema zurück
- `ensure_collection()` validiert bestehende Collections gegen `VectorSchema`
- bei Mismatch wird mit erwartetem und gefundenem Schema abgebrochen

### Memory

[src/adapters/vectordb/memory.py](../src/adapters/vectordb/memory.py) ist ein test-only Backend.

Es dient dazu, den Core unabhängig von Qdrant zu testen:

- gleiches Interface
- gleiches Schema-Verhalten
- keine externe Abhängigkeit

## 9. `doctor` als Preflight

Die CLI in [src/__main__.py](../src/__main__.py) hat jetzt ein eigenes `doctor`-Subcommand.

```bash
python -m src doctor --config config.yaml
```

`doctor` prüft:

- ob aktive Adapter importierbar sind
- ob optionale Drittbibliotheken korrekt geladen werden
- ob die Runtime instanziierbar ist
- ob das Backend erreichbar ist
- ob vorhandene Collections zum erwarteten Layout passen

Das ist der empfohlene erste Schritt nach jeder Config-Änderung.

## 10. Section-Aware Chunking im Detail

Das Verhalten von `section_aware` ist jetzt:

- gleiche `section_path` + gleiche Seite -> Blöcke werden bis `chunk_size` zusammengeführt
- Wechsel der `section_path` -> neuer Chunk
- Wechsel der Seite -> neuer Chunk
- ein einzelner zu langer Block wird weiterhin mit Overlap gesplittet

Traceability-Regeln:

- Merge-Chunk: `child_ids` enthält alle Ursprungsblöcke
- Merge-Chunk: `bbox = None`
- Split-Chunk: `parent_id` verweist auf den ursprünglichen Block

## 11. Pipeline-Robustheit

Die Pipelines validieren jetzt hart, dass die Anzahl der Embeddings exakt zur Anzahl der Elemente passt. Eine Teilindizierung mit fehlenden oder zu vielen Vektoren ist nicht mehr möglich.

Zusätzlich wurde das Bildhandling im Structured-Pfad gehärtet:

- Figure-Bilder werden sauber persistiert
- PIL-Dateien werden über Context Manager geöffnet

## 12. Manifest und Storage

Jeder Run schreibt ein Manifest mit:

- verwendeten Tools
- Tool-Versionen
- Counts
- Backend
- Namespace
- Collections
- Adapter-Signaturen
- Vector-Schemata

`document.json` enthält dieselben Layout-Metadaten pro Run, damit die Herkunft eines Index-Schreibvorgangs später nachvollziehbar bleibt.

## 13. Retrieval

`UnifiedRetriever` in [src/core/retrieval.py](../src/core/retrieval.py) arbeitet bewusst sequentiell:

1. Visual-Query
2. Text-Query
3. Tables-Query
4. Formulas-Query
5. Figures-Query
6. Fusion

Das ist absichtlich einfacher und robuster als parallele Query-Orchestrierung.

## 14. Environment-Härtung

Die Requirements-Struktur bleibt erhalten:

- `requirements.txt`
- `requirements-dev.txt`
- `requirements-amd.txt`
- `requirements-gpu.txt`

Neu dazu kommt [constraints-common.txt](../constraints-common.txt) für fragile Transitiv-Abhängigkeiten wie:

- `pdfplumber` / `pdfminer.six` / `cryptography`
- `qdrant-client` / `httpx` / `httpcore`

Dadurch ist die lokale Umgebung reproduzierbarer und unkontrollierte Transitiv-Upgrades werden abgefangen.

## 15. Qualitätsgrenzen

In [pyproject.toml](../pyproject.toml) sind die Quality-Gates jetzt realistischer zugeschnitten:

- `ruff`: Fokus auf Fehler, Importe und Konsistenz im stabilen Kern
- `mypy`: Fokus auf den stabilen Kern statt auf alle modellnahen Adapter

## 16. Praktischer Workflow

Bei einem Modell- oder Backend-Wechsel ist der empfohlene Ablauf:

1. `config.yaml` anpassen
2. `python -m src doctor --config config.yaml`
3. `python -m src ingest ...`
4. Retrieval gegen das aufgelöste Layout aufbauen

Wichtig: Wenn `index_namespace` nicht `legacy` ist, sollten Retrieval-Collections immer aus `resolve_index_layout(...)` kommen und nicht hart codiert sein.
