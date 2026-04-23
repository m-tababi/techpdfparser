# Staged Extraction Pipeline Design

## Ziel

Den Extraction-Block von einer Monolith-Pipeline (ein PDF → alle Rollen
in einem Prozess) auf eine Stage-basierte Pipeline umstellen, bei der
jede Rolle in einem eigenen OS-Prozess läuft und alle PDFs eines Batches
durch dieselbe Stage gehen, bevor die nächste Stage startet.

Das Ergebnis ist:

- **Strukturell OOM-frei.** Zwischen Stages endet der OS-Prozess und
  gibt die GPU komplett frei. Keine Abhängigkeit von PyTorch-Refcounts
  oder MinerU-Singleton-Tricks.
- **Batch-effizient.** Jedes Modell (MinerU, olmOCR-2-7B, Qwen2.5-VL-7B)
  lädt genau einmal pro Stage, nicht einmal pro PDF.
- **Resumable.** Jedes PDF trägt pro Stage einen Marker im Dateisystem.
  Ein Crash in Stage 2 bei PDF 237 kostet nur dessen Arbeit; Stage 1
  bleibt intakt, die übrigen PDFs in Stage 2 laufen weiter.
- **Nachvollziehbar für den Benutzer.** Die Stages werden manuell
  gestartet. Am Ende jeder Stage druckt das Kommando den exakten
  nächsten Befehl zum Kopieren. Die README listet alle vier Stages
  linear.

**Nicht im Scope:**

- Automatisches Verketten der Stages durch einen Python-Orchestrator
  (kein `batch`-Supercommand).
- Zeitstempel-Unterverzeichnisse für mehrere Extraktionen desselben
  PDFs. Re-Run eines PDFs bedeutet: Marker löschen und Stage neu
  starten.
- Persistente Worker-Prozesse (Modell bleibt zwischen Batches geladen).
  Falls später gebraucht, baut das auf denselben Stage-Grenzen auf.
- Änderungen am Output-Format. `content_list.json`, `document_rich.json`
  und die Sidecars bleiben bit-identisch zum heutigen Monolith-Run.

## Auslöser

Der aktuelle Stand verliert auf einer 24-GiB-GPU (21.4 GiB nutzbar nach
VDI-Overhead) wiederholt gegen CUDA-OOM:

- MinerU 2.5 cachet seine Modelle (Layout, OCR-det/rec, Table, MFR) in
  einem modulweiten Singleton (`mineru.backend.pipeline.pipeline_analyze.
  ModelSingleton._models`). Der Segmenter-Adapter hat keine Unload-API;
  `release_runtime_resources()` ruft nur `gc.collect()` +
  `torch.cuda.empty_cache()` auf, was allokierte Gewichte nicht freigibt.
- Beim anschließenden Load von olmOCR-2-7B (ca. 14 GiB bf16) bleibt
  MinerU resident. Gesamtbelegung erreicht > 17 GiB, der VDI-Agent
  (BlastWorker) hält konstant 2.3 GiB. Aktivierungen / KV-Cache während
  `model.generate()` überschreiten die verbleibenden 4 GiB.
- Der bestehende Role-Release-Fix (Commit `68f4425`, "feat(extraction):
  release GPU state between roles") greift strukturell nur zwischen
  Extractor-Rollen (olmOCR → Qwen), nicht zwischen Segmenter und erstem
  Extractor, weil MinerU keine koopperative Unload-API hat.
- Jeder Einzelfix (Env-Var `expandable_segments`, MinerU-Singleton-Leerung
  per Privat-API, `max_new_tokens` reduzieren) behandelt ein Symptom
  und bleibt fragil. Der Batch-Use-Case (viele PDFs, viele Seiten)
  verstärkt das Problem, weil jeder Lauf kumulative GPU-Fragmentierung
  anhäuft.

Stabilität vor Komfort: ein struktureller Umbau ist billiger als die
nächste Runde Flicks.

## Design-Entscheidungen

### 1. Vier Stages, vier Prozesse

| Stage | Kommando                      | Lädt            | Liest                             | Schreibt                                                          |
|-------|-------------------------------|-----------------|-----------------------------------|-------------------------------------------------------------------|
| 1     | `segment <pdf>...`            | MinerU, PyMuPDF | PDF-Dateien                       | `pages/<n>/page.png`, `segmentation.json` (inkl. Metadaten), Element-Sidecars für Passthrough-Rollen (heute: Tables) |
| 2     | `extract-text <outdir>...`    | olmOCR-2-7B     | `segmentation.json` + Page-PNGs   | Element-Sidecars für `TEXT`/`HEADING`                             |
| 3     | `describe-figures <outdir>...`| Qwen2.5-VL-7B   | `segmentation.json` + Page-PNGs   | Element-Sidecars für `FIGURE`/`DIAGRAM`/`TECHNICAL_DRAWING` + Visual-Crops |
| 4     | `assemble <outdir>...`        | nichts (CPU)    | alle Sidecars + `segmentation.json` | `content_list.json` (wie heute; `document_rich.json` ist Phase-2-Thema und bleibt ungeschrieben) |

Die Argumente sind **variadisch**: eine Stage verarbeitet beliebig viele
PDFs bzw. Output-Ordner in einer Schleife und beendet sich danach.

**Sidecar-Verantwortung: eine Region, genau eine schreibende Stage.**
Jede Stage ruft für ihre Region-Types `OutputWriter.write_element_sidecar(el)`
auf und schreibt ein komplettes `Element` (ID, bbox, content, etc.) nach
`pages/<page>/<element_id>_<type>.json` — exakt so wie die heutige
Monolith-Pipeline. Die Zuordnung der Region-Types zu Stages folgt der
Config: Stage 1 schreibt für jede Region, bei der der Role-Tool-Name dem
Segmenter-Namen entspricht (Passthrough — im Default also Tables), Stage 2
für `TEXT`/`HEADING`, Stage 3 für die Visual-Region-Types. Formula-Regionen
folgen dem aktuellen Verhalten: bei `formula_extractor = noop` werden sie
gedroppt; ein separater Formula-Extractor würde eine 5. Stage erfordern
(out-of-scope dieser Spec).

Stage 4 schreibt **keine** Element-Sidecars, sondern liest alle vorhandenen
mit `OutputWriter.read_all_sidecars()` ein und baut daraus `content_list.json`
und `document_rich.json` — identisch zum heutigen `rebuild`-Pfad.

**Config-Flag.** Alle vier Subcommands akzeptieren `--config <yaml-path>`
mit demselben Resolve-Verhalten wie heute `__main__.py:55-61`: explizit
übergeben > `extraction_config.yaml` im cwd > interne Defaults. Die Config
wird pro Stage-Aufruf frisch gelesen; es gibt keinen gemeinsamen Laufzeit-
zustand zwischen Stages.

**Lazy Model Load.** Eine Stage lädt ihr Modell erst, wenn mindestens ein
Path in der Argumentliste tatsächlich Arbeit erfordert (weder `.done` noch
vorheriger Marker fehlt). Wenn alle Pfade übersprungen werden, beendet
sich die Stage ohne Checkpoint-Load. Das spart bei inkrementellen Re-Runs
den 14 s olmOCR-Checkpoint bzw. 7 s MinerU-Init.

Stage 2/3/4 verweigern die Arbeit an einem Ordner, dessen vorheriger
Stage-Marker fehlt, mit einer klaren Fehlermeldung. Das verhindert
halbfertige Outputs, die später schwer zu diagnostizieren sind.

**Kein Fallback-Monolith.** Der heutige `extract`-Befehl wird entfernt,
nicht behalten. Zwei Modi parallel zu pflegen ist teuer und der Grund
für genau die Fragilität, die wir loswerden wollen.

### 2. Output-Basis ist fest, Ordnername kommt vom PDF

Der Base-Pfad ist `outputs/`. Kein `--output`-Flag. Stage 1 legt pro PDF
einen Unterordner an, dessen Name aus dem PDF-Dateinamen ohne Endung
abgeleitet wird:

```
outputs/jmmp-09-00199-v2/
outputs/3_HRA_for_offshore/
```

Wenn ein Zielordner schon Artefakte enthält und der passende Stage-Marker
fehlt, bricht die Stage für diesen einen Ordner ab und schreibt `.error`.
Der Benutzer löscht den Ordner bewusst und startet neu.

**Keine Zeitstempel.** Sie machen Pfade unhandlich und erzeugen Gräber
alter Läufe, die der Benutzer selbst aufräumen muss.

### 3. Stage-Marker als Fortschritts-State

Struktur pro PDF-Ordner:

```
outputs/<pdf-stem>/.stages/
├── segment.done            oder segment.error
├── extract-text.done       oder extract-text.error
├── describe-figures.done   oder describe-figures.error
└── assemble.done           oder assemble.error
```

- `.done` wird **am Ende** einer erfolgreichen Stage geschrieben (nach
  allen Sidecars, nicht inkrementell). Leere Datei; Existenz ist die
  Information.
- `.error` enthält Traceback + relevanten Kontext (PDF-Pfad,
  Konfiguration, Stage-Name, Python-Umgebung). Erfolg und Fehler
  schließen sich gegenseitig aus: ein erfolgreicher Re-Run löscht
  `.error` und schreibt `.done`.
- Stage-Logik pro Path in der Argumentliste in dieser Reihenfolge:
  1. Eigener `.done`-Marker existiert → überspringen.
  2. Vorgänger-Stage-Marker fehlt (nur Stage 2/3/4) → `.error`
     schreiben, weiter mit dem nächsten Path.
  3. Sonst verarbeiten, am Ende `.done` schreiben.

  Punkt 1 vor Punkt 2: wenn der eigene `.done` schon da ist, ist die
  Arbeit erledigt, unabhängig davon, ob jemand zwischendurch den
  Vorgänger-Marker manuell weggeräumt hat.

Das Verzeichnis heißt `.stages` (mit führendem Punkt), weil es
Projekt-interner Zustand ist und in normalen `ls`-Listings nicht
zuoberst stehen soll.

### 4. Fehlerbehandlung

Pro-PDF-Isolation innerhalb einer Stage:

```python
for path in paths:
    if writer.is_stage_done(stage_name):
        continue
    try:
        process(path)
        writer.mark_stage_done(stage_name)
    except Exception as exc:
        writer.write_stage_error(stage_name, exc)
        continue
```

Sichtbarkeit des Fehlers:

- **Inline während der Stage läuft:** jede Zeile druckt `✓` oder `✗`
  plus eine kurze Fehlerzeile + Pfad zur `.error`-Datei.
- **Block am Ende:** deutlich abgegrenzte Zusammenfassung mit Anzahl
  erfolgreicher und fehlgeschlagener PDFs, Liste mit `✓`/`✗` je Ordner,
  und — wichtig — der **nächste Befehl enthält nur die erfolgreichen
  Ordner.** Der Benutzer kann ihn blind kopieren, ohne manuell kaputte
  Pfade rauszufiltern.
- **Exit-Code:** `0` wenn alle PDFs im aktuellen Aufruf erfolgreich oder
  via Marker übersprungen wurden, sonst `1`.

CUDA-OOM-Fallback auf Adapter-Ebene (`olmocr2_text.py:81-98`,
entsprechend in `qwen25vl_figure.py`) bleibt unverändert: einzelne
Seiten, die auf GPU nicht passen, fallen auf CPU zurück und erzeugen
eine `RuntimeWarning`. Das ist die letzte Reißleine innerhalb eines
PDFs und passt zur strukturellen Pro-Prozess-Isolation zwischen Stages.

### 5. Rückmeldung nach jeder Stage

Format am Ende jeder Stage, exemplarisch:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Stage 'segment': 2 erfolgreich, 1 FEHLGESCHLAGEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ outputs/jmmp-09-00199-v2       (19 Seiten, 127 Regions)
  ✗ outputs/3_HRA_for_offshore     (Fehler: siehe .stages/segment.error)
  ✓ outputs/kleines_doc            (8 Seiten, 43 Regions)

Nächster Schritt (nur erfolgreiche Ordner):
  python -m extraction extract-text outputs/jmmp-09-00199-v2 outputs/kleines_doc
```

Die letzte Stage (`assemble`) druckt statt des nächsten Befehls die
Pfade der erzeugten `content_list.json`-Dateien als Abschluss-Signal.

### 6. README

Die bestehende `## CLI`-Sektion wird entfernt und durch
`## Extraction-Pipeline` ersetzt. Neue Sektion enthält in linearer
Reihenfolge vier nummerierte Unterabschnitte (Stage 1 bis 4), jeweils
mit einer einzeiligen Beschreibung und dem Kommando. Darunter ein
Abschnitt "Einzelnen Schritt neu laufen lassen" mit der Marker-Delete-
Operation als Beispiel.

**Unberührt bleiben:** `## Install`, `## CPU-only config example` (die
Adapter-Namen `pymupdf`, `pymupdf_text`, `noop` sind alle registriert
und korrekt), `## Quality gates`. Diese Abschnitte wurden gegen den
aktuellen Code (`pyproject.toml`, `extraction/adapters/*`,
`extraction/registry.py`) geprüft und stimmen.

### 7. Migration

| Datei                              | Änderung                                                                 |
|------------------------------------|--------------------------------------------------------------------------|
| `extraction/__main__.py`           | `extract` und `rebuild` raus. Vier neue Subcommands.                     |
| `extraction/pipeline.py`           | `ExtractionPipeline.run()` raus. Vier Funktionen `run_segment()`, `run_text()`, `run_figures()`, `run_assemble()`. |
| `extraction/output.py`             | `OutputWriter` bekommt `mark_stage_done()`, `is_stage_done()`, `write_stage_error()`. `write_segmentation()` bekommt zusätzliche Parameter `doc_id`, `source_file`, `total_pages`, `segmentation_tool` und schreibt diese als Top-Level-Felder neben `regions`. `read_segmentation(output_dir)` liefert alle vier Metadatenfelder + Region-Liste für Stage 2/3/4 zurück. |
| `extraction/config.py`             | Konstante `DEFAULT_OUTPUT_BASE = "outputs"`.                             |
| `extraction/adapters/*`            | Unverändert.                                                             |
| `extraction/_runtime.py`           | Bleibt; `release_runtime_resources()` wird nicht mehr aus der Pipeline gerufen, `is_cuda_oom()` bleibt für den Adapter-internen CPU-Fallback. |
| `extraction/tests/`                | Tests pro Stage mit Stubs; ein Integrations-Test durch alle vier Stages. |

**Explizit unangetastet:**

- Output-Format (`content_list.json`, Sidecars). Das ist der Vertrag
  zur Embedding-Stage und muss stabil bleiben. `document_rich.json`
  ist in der heutigen Pipeline nicht verdrahtet (Phase-2-Thema) und
  wird auch von Stage 4 nicht geschrieben.
- Das YAML-Config-Schema und alle Adapter-Config-Blöcke.
- `docs/extraction_output.md`, `docs/principles.md`,
  `docs/architecture.md`. Falls nach dem Umbau ein Nachtrag zur
  Pipeline-Struktur in `architecture.md` sinnvoll ist, wird er separat
  committed, nicht als Teil dieses Specs.

**Breaking Change, aber interner Zwischenzustand:** Die Struktur von
`segmentation.json` ändert sich von `list[Region]` zu
`{doc_id, source_file, total_pages, segmentation_tool, regions: list[Region]}`.
`segmentation.json` ist kein externer Kontrakt, sondern Inter-Stage-State;
der Schema-Wechsel wird nicht versioniert, bestehende Outputs aus dem
alten Monolith-Lauf müssen neu erzeugt werden.

### 8. Testing

- **Unit pro Stage**, mit Stubs aus `extraction/adapters/stubs.py`:
  - Happy Path: Marker wird geschrieben, Output-Dateien liegen richtig.
  - Skip-Verhalten: Marker existiert → Arbeit wird nicht wiederholt,
    Adapter wird nicht instanziiert.
  - Error-Path: Stub wirft Exception → `.error` wird geschrieben,
    andere Pfade in der Argumentliste laufen weiter, Exit-Code 1.
  - Vorgänger-Marker fehlt: Stage bricht für diesen Ordner ab mit
    klarer Meldung und Exit-Code 1.
- **Integrationstest** (`@pytest.mark.integration`): ein kurzes PDF
  wird durch alle vier Stages geschleust. `content_list.json` wird mit
  einem vorher gespeicherten Referenz-Output des heutigen Monolith-`run()`
  verglichen. Die Prüfung ist **zweigeteilt**, weil die VLM-Adapter
  stochastisch sind (`olmocr2_text.py:136-142` nutzt
  `do_sample=True, temperature=0.1`; Qwen2.5-VL ebenso):

  - **Strikt bit-identisch:** `elements[i].element_id`, `page`, `bbox`,
    `type`, `reading_order_index`, `confidence`, `extractor`,
    `content.markdown`, `content.latex`, `content.caption` (letztere drei
    kommen aus dem deterministischen MinerU-Pfad). Ebenso
    `pages[].element_ids` und das globale `reading_order_index` aus
    `build_content_list()`.
  - **Strukturell geprüft:** `content.text` (olmOCR) und
    `content.description` (Qwen) — Feld muss vorhanden und nicht-leer
    sein; Längen-/Typ-Plausibilität reicht. Eine exakte Gleichheit
    würde deterministisches Seeding der Sampling-Aufrufe verlangen,
    das heute in den Adaptern nicht existiert und auch nicht
    eingeführt werden soll (keine Änderung an Adapter-Verhalten im
    Scope dieser Spec).

  Das ist das zentrale Safety-Net: der strukturelle Teil des
  Ausgabe-Formats (Schema, IDs, Ordering) darf sich nicht verschieben,
  sonst bricht der nachgelagerte Embedding-Block.

## Offene Punkte

Keine. Alle Design-Fragen wurden im Brainstorming geklärt:

- Kein Fallback-Monolith.
- `outputs/` als feste Basis, Ordnername vom PDF-Dateinamen.
- Kein Zeitstempel.
- Stage-Namen: `segment`, `extract-text`, `describe-figures`, `assemble`.
- Fehlerverhalten: weitermachen bei Einzelfehlern, sichtbarer Block am
  Ende, Exit-Code 1 bei mindestens einem Fehler.
- README: neuer `## Extraction-Pipeline`-Block ersetzt `## CLI`; Rest
  bleibt.
