# techpdfparser

Technical-PDF parsing into a unified structured output: text, tables,
formulas, figures, diagrams, drawings. Output format is a stable contract
(see `docs/extraction_output.md`). Tools are swappable per role via YAML.

## Install

GPU-only. Voraussetzung: NVIDIA-GPU mit aktuellem CUDA-Treiber.
Empfohlen: ein lokales venv **außerhalb** des Projekt-Ordners —
besonders auf Netzlaufwerken, wo venvs unzuverlässig sind.

**Reihenfolge ist wichtig:** PyTorch (Schritt 2) muss **vor** dem
Projekt-Install (Schritt 3) kommen. Sonst zieht pip die CPU-Wheels
von PyPI und du hast ohne es zu merken kein CUDA.

### 1. Venv anlegen und aktivieren

    python3 -m venv ~/venvs/phase1
    source ~/venvs/phase1/bin/activate
    pip install --upgrade pip

### 2. PyTorch vom CUDA-Wheel-Index installieren

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

(Andere CUDA-Builds: https://pytorch.org/get-started/locally/)

### 3. Projekt installieren — ein Befehl, alles drin

    pip install -e ".[dev]"

Bringt alles in einem Schritt: MinerU, transformers (mit `<5`-Cap,
siehe Hinweis), accelerate, beautifulsoup4, MinerU-Transitiv-Deps,
**alle Tabellen-Extraktoren** (TATR, docling, qwen25vl_table) plus
die Dev-Tools (pytest, ruff, mypy). Lass `[dev]` weg, wenn du nur
die Pipeline laufen lassen willst — dann reicht `pip install -e .`.

### 4. Tesseract-System-Binary (nur für TATR und docling_table)

    sudo apt install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng

Nur nötig, wenn du `configs/tatr_table.yaml` oder
`configs/docling_table.yaml` benutzt — beide rufen Tesseract für die
Zell-OCR auf. Standard-Lauf und `configs/qwen25vl_table.yaml` brauchen
das nicht (Qwen2.5-VL macht Struktur + OCR in einem Schritt).

### 5. Verifizieren

    pip list | grep -iE "torch|mineru|transformers|accelerate|techpdfparser"

Erwartet:

    accelerate     1.13.x
    mineru         3.1.x
    techpdfparser  0.1.0   <projekt-pfad>   # editable
    torch          2.11.x+cu130
    torchvision    0.26.x+cu130
    transformers   4.5x.y                   # 4.x wegen <5-Cap

### Hinweis: `transformers<5` Cap

Der olmOCR-2-Adapter (`extraction/adapters/olmocr2_text.py`) importiert
`AutoModelForVision2Seq`, das in transformers 5.0 entfernt wurde
(Ersatz: `AutoModelForImageTextToText`). Wir pinnen deshalb
`transformers<5` in den Dependencies, bis der Adapter migriert ist.

## Extraction-Pipeline

Die Extraktion läuft in vier manuellen Schritten. Jedes Kommando lädt
das zugehörige Modell, verarbeitet alle genannten PDFs/Ordner, und
beendet sich (gibt die GPU wieder frei). Am Ende jedes Schritts wird
der nächste Befehl zum Kopieren ausgegeben.

### 1. Segmentieren (MinerU)

Input: PDF-Pfade. Legt Output-Ordner automatisch unter `outputs/` an.

    python -m extraction segment <pdf1> <pdf2> ...

### MinerU-Backend wechseln

Der MinerU-Adapter ist unter drei Registry-Namen registriert, einer pro
Backend:

- `mineru25`: `pipeline`
- `mineru_hybrid`: `hybrid-auto-engine`
- `mineru_vlm`: `vlm-auto-engine`

Welcher aktiv ist, steht in `extraction/config.py` bzw. der genutzten
YAML-Config.

Hybrid testen:

    extraction:
      segmenter: mineru_hybrid
      text_extractor: mineru_hybrid
      table_extractor: mineru_hybrid
      formula_extractor: mineru_hybrid

VLM testen:

    extraction:
      segmenter: mineru_vlm
      text_extractor: mineru_vlm
      table_extractor: mineru_vlm
      formula_extractor: mineru_vlm

Ein Backend-Wechsel ändert die gespeicherte `stage_config`. Bestehende
Output-Ordner werden deshalb beim erneuten `segment` bewusst als stale
markiert; nutze für Vergleiche einen frischen Output-Ordner. Für Hybrid/VLM
kann je nach MinerU-Installation eine vollständige Installation wie
`mineru[all]` nötig sein.

### 2. Text, Tabellen und Formeln ergänzen

Wenn `text_extractor`, `table_extractor` und `formula_extractor` auf
denselben Adapter wie der Segmenter zeigen (z. B. alle auf einen der
MinerU-Adapter), hat `segment` die Inhalte bereits als Sidecars
geschrieben — `extract-text` überspringt sie dann. Setzt die Config einen
abweichenden Extractor (z. B. `text_extractor: olmocr2` oder
`table_extractor: qwen25vl_table`), läuft die jeweilige Extraktion hier
über die gespeicherten Crops.

    python -m extraction extract-text <outdir1> <outdir2> ...

### 3. Figures beschreiben (Qwen2.5-VL)

    python -m extraction describe-figures <outdir1> <outdir2> ...

### 4. Zusammenbauen (CPU, kein Modell)

    python -m extraction assemble <outdir1> <outdir2> ...

## Run-Sicherheit

`segment` schreibt Run-Metadaten nach `segmentation.json`: PDF-Hash,
Source-Datei, effektive Render-DPI und die relevante Stage-Config. Wenn
`segment.done` existiert, aber diese Metadaten nicht mehr zur aktuellen
PDF/Config passen, bricht die Stage mit `.stages/segment.error` ab.
Vorhandene Artefakte werden nicht automatisch gelöscht oder überschrieben.

Die späteren Stages verwenden beim Cropping immer die in
`segmentation.json` gespeicherte `render_dpi`. So bleiben Crops korrekt,
auch wenn die aktuelle Config inzwischen eine andere DPI enthält.

### Einzelnen Schritt neu laufen lassen

Stage-Marker verhindern komplette Wiederholungen bereits erledigter Schritte.
`extract-text` überspringt zusätzlich vorhandene Sidecars, damit
MinerU-Passthrough-Inhalte aus `segment` erhalten bleiben. Für eine bewusste
Neu-Extraktion derselben Zielartefakte gibt es `--force`:

    python -m extraction extract-text --force outputs/jmmp-09-00199-v2
    python -m extraction describe-figures --force outputs/jmmp-09-00199-v2

`--force` überschreibt nur die Sidecars/Crops der jeweiligen Stage und
entfernt `assemble.done`; danach `assemble` erneut ausführen. Wenn die
Config einen MinerU-Passthrough nutzt (Segmenter und Extractor mit
demselben Registry-Namen), ist `extract-text --force` normalerweise
nicht nötig, weil die Inhalte schon aus `segment` stammen.

Tabellen und Formeln dürfen als Crop-only Fallback persistieren, wenn der
Extractor keinen strukturierten Inhalt liefert. In diesem Fall bleibt
mindestens `content.image_path` als visuelle Evidenz erhalten.

## CPU-only config example

    extraction:
      renderer: pymupdf
      segmenter: pymupdf_text
      text_extractor: noop
      table_extractor: noop
      formula_extractor: noop
      figure_descriptor: noop
      output_dir: outputs
      dpi: 150

## Quality gates

    pytest -q
    ruff check extraction
    mypy

Integration tests (MinerU / GPU) are marker-gated:

    pytest -m integration
