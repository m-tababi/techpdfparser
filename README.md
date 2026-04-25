# techpdfparser

Technical-PDF parsing into a unified structured output: text, tables,
formulas, figures, diagrams, drawings. Output format is a stable contract
(see `docs/extraction_output.md`). Tools are swappable per role via YAML.

## Install

GPU stack: CUDA build of PyTorch + MinerU 3.1+ (segmenter), olmOCR-2
(text), Qwen2.5-VL (figures). Recommended: a local venv **outside**
the project directory — especially on network drives, where venvs
are unreliable.

### 1. Venv anlegen und aktivieren

    python3 -m venv ~/venvs/phase1
    source ~/venvs/phase1/bin/activate
    pip install --upgrade pip

### 2. PyTorch vom CUDA-13-Wheel-Index installieren

`torch` und `torchvision` müssen vom PyTorch-Index kommen — PyPI
liefert nur CPU-Wheels.

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

(Andere CUDA-/CPU-Builds: https://pytorch.org/get-started/locally/)

### 3. Projekt + Extras installieren

    pip install -e ".[gpu,dev]"

`[gpu]` zieht MinerU, transformers (mit `<5`-Cap, siehe Hinweis),
torchvision, accelerate, beautifulsoup4. `[dev]` zieht pytest,
pytest-cov, ruff, mypy.

### 4. Verifizieren

    pip list | grep -iE "torch|mineru|transformers|accelerate|techpdfparser"

Erwartet:

    accelerate     1.13.x
    mineru         3.1.x
    techpdfparser  0.1.0   <projekt-pfad>   # editable
    torch          2.11.x+cu130
    torchvision    0.26.x+cu130
    transformers   4.5x.y                   # 4.x wegen <5-Cap

### CPU-only Fallback

    pip install -e .

Nur der `pymupdf_text`-Segmenter und die `noop`-Adapter funktionieren —
kein MinerU, keine VLM-Extraktion.

### Hinweis: `transformers<5` Cap

Der olmOCR-2-Adapter (`extraction/adapters/olmocr2_text.py`) importiert
`AutoModelForVision2Seq`, das in transformers 5.0 entfernt wurde
(Ersatz: `AutoModelForImageTextToText`). `[gpu]` pinnt deshalb
`transformers<5`, bis der Adapter migriert ist.

## Extraction-Pipeline

Die Extraktion läuft in vier manuellen Schritten. Jedes Kommando lädt
das zugehörige Modell, verarbeitet alle genannten PDFs/Ordner, und
beendet sich (gibt die GPU wieder frei). Am Ende jedes Schritts wird
der nächste Befehl zum Kopieren ausgegeben.

### 1. Segmentieren (MinerU)

Input: PDF-Pfade. Legt Output-Ordner automatisch unter `outputs/` an.

    python -m extraction segment <pdf1> <pdf2> ...

### 2. Text extrahieren (olmOCR-2)

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

Standardmäßig überspringt eine Stage bereits vorhandene Sidecars. Für eine
bewusste Neu-Extraktion derselben Zielartefakte gibt es `--force`:

    python -m extraction extract-text --force outputs/jmmp-09-00199-v2
    python -m extraction describe-figures --force outputs/jmmp-09-00199-v2

`--force` überschreibt nur die Sidecars/Crops der jeweiligen Stage und
entfernt `assemble.done`; danach `assemble` erneut ausführen.

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
