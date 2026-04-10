"""CLI entrypoint: python -m techpdfparser ingest <pdf> [--config config.yaml]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import src.adapters  # noqa: F401 — triggers all @register_* decorators

from src.core.config import AppConfig, default_config, get_adapter_config, load_config
from src.core.models.document import DocumentMeta
from src.core.pipelines.structured import StructuredPipeline
from src.core.pipelines.text import TextPipeline
from src.core.pipelines.visual import VisualPipeline
from src.core.registry import (
    get_figure_descriptor,
    get_formula_extractor,
    get_renderer,
    get_structured_parser,
    get_text_chunker,
    get_text_embedder,
    get_text_extractor,
    get_visual_embedder,
    get_index_writer,
)
from src.utils.ids import generate_doc_id
from src.utils.storage import StorageManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="techpdfparser")
    sub = parser.add_subparsers(dest="command")

    ingest = sub.add_parser("ingest", help="Ingest a PDF into all three pipelines")
    ingest.add_argument("pdf", type=Path, help="Path to the PDF file")
    ingest.add_argument("--config", type=Path, default=None, help="Path to config YAML")

    return parser.parse_args()


def _load_cfg(config_path: Path | None) -> AppConfig:
    if config_path is not None:
        return load_config(config_path)
    return default_config()


def _build_doc_meta(pdf_path: Path, renderer) -> DocumentMeta:
    doc_id = generate_doc_id(str(pdf_path.resolve()))
    total_pages = renderer.page_count(pdf_path)
    return DocumentMeta(
        doc_id=doc_id,
        source_file=str(pdf_path.resolve()),
        total_pages=total_pages,
        file_size_bytes=pdf_path.stat().st_size,
    )


def _run_ingest(pdf_path: Path, cfg: AppConfig) -> None:
    storage = StorageManager(cfg.storage.base_dir)

    vc = cfg.pipelines.visual
    tc = cfg.pipelines.text
    sc = cfg.pipelines.structured

    renderer = get_renderer(vc.renderer, **get_adapter_config(cfg, vc.renderer))
    doc_meta = _build_doc_meta(pdf_path, renderer)
    print(f"Ingesting {pdf_path.name} | doc_id={doc_meta.doc_id} | pages={doc_meta.total_pages}")

    index_writer = get_index_writer("qdrant", **get_adapter_config(cfg, "qdrant"))

    visual_pipeline = VisualPipeline(
        renderer=renderer,
        embedder=get_visual_embedder(vc.embedder, **get_adapter_config(cfg, vc.embedder)),
        index_writer=index_writer,
        storage=storage,
        config=vc,
    )

    text_pipeline = TextPipeline(
        extractor=get_text_extractor(tc.extractor, **get_adapter_config(cfg, tc.extractor)),
        chunker=get_text_chunker(tc.chunker, **get_adapter_config(cfg, tc.chunker)),
        embedder=get_text_embedder(tc.embedder, **get_adapter_config(cfg, tc.embedder)),
        index_writer=index_writer,
        storage=storage,
        config=tc,
    )

    structured_pipeline = StructuredPipeline(
        parser=get_structured_parser(sc.parser, **get_adapter_config(cfg, sc.parser)),
        formula_extractor=get_formula_extractor(
            sc.formula_extractor, **get_adapter_config(cfg, sc.formula_extractor)
        ),
        figure_descriptor=get_figure_descriptor(
            sc.figure_descriptor, **get_adapter_config(cfg, sc.figure_descriptor)
        ),
        embedder=get_text_embedder(tc.embedder, **get_adapter_config(cfg, tc.embedder)),
        index_writer=index_writer,
        storage=storage,
        config=sc,
        renderer=renderer,
    )

    pages = visual_pipeline.run(pdf_path, doc_meta)
    print(f"  Visual:     {len(pages)} pages indexed")

    chunks = text_pipeline.run(pdf_path, doc_meta)
    print(f"  Text:       {len(chunks)} chunks indexed")

    tables, formulas, figures = structured_pipeline.run(pdf_path, doc_meta)
    print(f"  Structured: {len(tables)} tables, {len(formulas)} formulas, {len(figures)} figures indexed")


def main() -> None:
    args = _parse_args()

    if args.command is None:
        print("Usage: python -m techpdfparser ingest <pdf> [--config config.yaml]")
        sys.exit(1)

    cfg = _load_cfg(args.config)
    _run_ingest(args.pdf, cfg)


if __name__ == "__main__":
    main()
