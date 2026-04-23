# techpdfparser

Technical-PDF parsing into a unified structured output: text, tables,
formulas, figures, diagrams, drawings. Output format is a stable contract
(see `docs/extraction_output.md`). Tools are swappable per role via YAML.

## Install

CPU-only (no segmenter):

    pip install -e .

Full GPU stack (MinerU 2.5, OlmOCR, Qwen2.5-VL):

    pip install -e .[gpu]

PyTorch: CUDA builds depend on your system. Follow
https://pytorch.org/get-started/locally/ and install a matching
`torch` wheel before or alongside `-e .[gpu]`.

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

### Einzelnen Schritt neu laufen lassen

Marker löschen und Stage neu starten, z.B. Text-Extraktion für ein PDF:

    rm outputs/jmmp-09-00199-v2/.stages/extract-text.done
    python -m extraction extract-text outputs/jmmp-09-00199-v2

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
