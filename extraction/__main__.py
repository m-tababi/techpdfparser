"""CLI entrypoint: python -m extraction extract <pdf> [--config config.yaml]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import extraction.adapters  # noqa: F401 — trigger adapter registration

from .config import ExtractionConfig, load_extraction_config
from .pipeline import ExtractionPipeline
from .registry import (
    get_figure_descriptor,
    get_formula_extractor,
    get_renderer,
    get_segmenter,
    get_table_extractor,
    get_text_extractor,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="extraction")
    sub = parser.add_subparsers(dest="command")

    extract = sub.add_parser("extract", help="Extract structured content from a PDF")
    extract.add_argument("pdf", type=Path, help="Path to the PDF file")
    extract.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    extract.add_argument("--output", type=Path, default=None, help="Output directory")

    return parser.parse_args()


def _load_cfg(config_path: Path | None) -> ExtractionConfig:
    if config_path is not None:
        return load_extraction_config(config_path)
    default = Path("extraction_config.yaml")
    if default.exists():
        return load_extraction_config(default)
    return ExtractionConfig()


def _run_extract(pdf_path: Path, cfg: ExtractionConfig, output_dir: Path | None) -> None:
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    out = output_dir or Path(cfg.output_dir)

    renderer = get_renderer(cfg.renderer, **cfg.get_adapter_config(cfg.renderer))
    segmenter = get_segmenter(cfg.segmenter, **cfg.get_adapter_config(cfg.segmenter))
    text_extractor = get_text_extractor(
        cfg.text_extractor, **cfg.get_adapter_config(cfg.text_extractor)
    )
    table_extractor = get_table_extractor(
        cfg.table_extractor, **cfg.get_adapter_config(cfg.table_extractor)
    )
    formula_extractor = get_formula_extractor(
        cfg.formula_extractor, **cfg.get_adapter_config(cfg.formula_extractor)
    )
    figure_descriptor = get_figure_descriptor(
        cfg.figure_descriptor, **cfg.get_adapter_config(cfg.figure_descriptor)
    )

    pipeline = ExtractionPipeline(
        renderer=renderer,
        segmenter=segmenter,
        text_extractor=text_extractor,
        table_extractor=table_extractor,
        formula_extractor=formula_extractor,
        figure_descriptor=figure_descriptor,
        output_dir=out,
        confidence_threshold=cfg.confidence_threshold,
    )

    print(f"Extracting {pdf_path.name}...")
    content_list = pipeline.run(pdf_path)
    print(f"  Elements: {len(content_list.elements)}")
    print(f"  Pages:    {content_list.total_pages}")
    print(f"  Output:   {out}")


def main() -> None:
    args = _parse_args()

    if args.command is None:
        print("Usage: python -m extraction [extract] ...")
        sys.exit(1)

    cfg = _load_cfg(getattr(args, "config", None))

    if args.command == "extract":
        _run_extract(args.pdf, cfg, getattr(args, "output", None))
        return

    print(f"Unknown command: {args.command}")
    sys.exit(1)


if __name__ == "__main__":
    main()
