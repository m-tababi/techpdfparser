"""CLI entrypoint: python -m extraction {segment, extract-text, describe-figures, assemble}."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import extraction.adapters  # noqa: F401 — trigger adapter registration

from .config import ExtractionConfig, load_extraction_config
from .stages.assemble import run_assemble
from .stages.describe_figures import run_figures
from .stages.extract_text import run_text
from .stages.segment import run_segment


def _load_cfg(config_path: Path | None) -> ExtractionConfig:
    if config_path is not None:
        return load_extraction_config(config_path)
    default = Path("extraction_config.yaml")
    if default.exists():
        return load_extraction_config(default)
    return ExtractionConfig()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="extraction")
    sub = parser.add_subparsers(dest="command", required=True)

    seg = sub.add_parser("segment", help="Stage 1: render + segment PDFs")
    seg.add_argument("pdfs", nargs="+", type=Path)
    seg.add_argument("--config", type=Path, default=None)
    seg.add_argument("--out", type=Path, default=None)

    txt = sub.add_parser("extract-text", help="Stage 2: text extraction")
    txt.add_argument("outdirs", nargs="+", type=Path)
    txt.add_argument("--config", type=Path, default=None)

    fig = sub.add_parser("describe-figures", help="Stage 3: figure descriptions")
    fig.add_argument("outdirs", nargs="+", type=Path)
    fig.add_argument("--config", type=Path, default=None)

    asm = sub.add_parser("assemble", help="Stage 4: build content_list.json")
    asm.add_argument("outdirs", nargs="+", type=Path)
    asm.add_argument("--config", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(getattr(args, "config", None))

    if args.command == "segment":
        out_base = args.out if args.out is not None else Path(cfg.output_dir)
        sys.exit(run_segment(args.pdfs, cfg, out_base))
    if args.command == "extract-text":
        sys.exit(run_text(args.outdirs, cfg))
    if args.command == "describe-figures":
        sys.exit(run_figures(args.outdirs, cfg))
    if args.command == "assemble":
        sys.exit(run_assemble(args.outdirs, cfg))
    print(f"Unknown command: {args.command}")
    sys.exit(1)


if __name__ == "__main__":
    main()
