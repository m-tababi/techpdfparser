"""CLI entrypoint: python -m src ingest <pdf> [--config config.yaml]."""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import src.adapters  # noqa: F401 - triggers all @register_* decorators
from src.core.config import AppConfig, default_config, get_adapter_config, load_config
from src.core.indexing import (
    ResolvedIndexLayout,
    resolve_index_layout,
    schema_matches,
)
from src.core.models.document import DocumentMeta
from src.core.pipelines.structured import StructuredPipeline
from src.core.pipelines.text import TextPipeline
from src.core.pipelines.visual import VisualPipeline
from src.core.registry import (
    get_figure_descriptor,
    get_formula_extractor,
    get_fusion_engine,
    get_index_writer,
    get_renderer,
    get_retrieval_engine,
    get_structured_parser,
    get_text_chunker,
    get_text_embedder,
    get_text_extractor,
    get_visual_embedder,
)
from src.utils.ids import generate_doc_id
from src.utils.storage import StorageManager

_DEPENDENCY_MODULES: dict[str, tuple[str, ...]] = {
    "pymupdf": ("fitz",),
    "colqwen25": ("colpali_engine.models",),
    "clip": ("transformers",),
    "olmocr2": ("transformers",),
    "pymupdf_text": ("fitz",),
    "pymupdf_structured": ("fitz",),
    "bge_m3": ("FlagEmbedding",),
    "minilm": ("sentence_transformers",),
    "mineru25": ("mineru.cli.common",),
    "pdfplumber": ("pdfplumber",),
    "ppformulanet": ("paddleocr",),
    "pix2tex": ("pix2tex.cli",),
    "qwen25vl": ("transformers",),
    "moondream": ("transformers",),
    "qdrant": ("qdrant_client",),
    "memory": (),
    "rrf": (),
    "score_norm": (),
}


@dataclass
class ActiveRuntime:
    renderer: Any
    visual_embedder: Any
    text_extractor: Any
    text_chunker: Any
    text_embedder: Any
    structured_parser: Any
    formula_extractor: Any
    figure_descriptor: Any
    index_writer: Any
    retrieval_engine: Any
    fusion_engine: Any
    index_layout: ResolvedIndexLayout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src")
    sub = parser.add_subparsers(dest="command")

    ingest = sub.add_parser("ingest", help="Ingest a PDF into all three pipelines")
    ingest.add_argument("pdf", type=Path, help="Path to the PDF file")
    ingest.add_argument("--config", type=Path, default=None, help="Path to config YAML")

    doctor = sub.add_parser("doctor", help="Validate config, dependencies, and index layout")
    doctor.add_argument("--config", type=Path, default=None, help="Path to config YAML")

    return parser.parse_args()


def _load_cfg(config_path: Path | None) -> AppConfig:
    if config_path is not None:
        return load_config(config_path)
    return default_config()


def _build_runtime(cfg: AppConfig) -> ActiveRuntime:
    vc = cfg.pipelines.visual
    tc = cfg.pipelines.text
    sc = cfg.pipelines.structured
    backend_name = cfg.retrieval.retrieval_engine

    renderer = get_renderer(vc.renderer, **get_adapter_config(cfg, vc.renderer))
    visual_embedder = get_visual_embedder(
        vc.embedder, **get_adapter_config(cfg, vc.embedder)
    )
    text_extractor = get_text_extractor(
        tc.extractor, **get_adapter_config(cfg, tc.extractor)
    )
    text_chunker = get_text_chunker(tc.chunker, **get_adapter_config(cfg, tc.chunker))
    text_embedder = get_text_embedder(
        tc.embedder, **get_adapter_config(cfg, tc.embedder)
    )
    structured_parser = get_structured_parser(
        sc.parser, **get_adapter_config(cfg, sc.parser)
    )
    formula_extractor = get_formula_extractor(
        sc.formula_extractor, **get_adapter_config(cfg, sc.formula_extractor)
    )
    figure_descriptor = get_figure_descriptor(
        sc.figure_descriptor, **get_adapter_config(cfg, sc.figure_descriptor)
    )
    index_writer = get_index_writer(
        backend_name, **get_adapter_config(cfg, backend_name)
    )
    retrieval_engine = get_retrieval_engine(
        backend_name, **get_adapter_config(cfg, backend_name)
    )
    fusion_engine = get_fusion_engine(
        cfg.retrieval.fusion_engine,
        **get_adapter_config(cfg, cfg.retrieval.fusion_engine),
    )

    return ActiveRuntime(
        renderer=renderer,
        visual_embedder=visual_embedder,
        text_extractor=text_extractor,
        text_chunker=text_chunker,
        text_embedder=text_embedder,
        structured_parser=structured_parser,
        formula_extractor=formula_extractor,
        figure_descriptor=figure_descriptor,
        index_writer=index_writer,
        retrieval_engine=retrieval_engine,
        fusion_engine=fusion_engine,
        index_layout=resolve_index_layout(
            cfg,
            visual_embedder=visual_embedder,
            text_embedder=text_embedder,
        ),
    )


def _active_adapter_names(cfg: AppConfig) -> dict[str, str]:
    return {
        "renderer": cfg.pipelines.visual.renderer,
        "visual_embedder": cfg.pipelines.visual.embedder,
        "text_extractor": cfg.pipelines.text.extractor,
        "text_chunker": cfg.pipelines.text.chunker,
        "text_embedder": cfg.pipelines.text.embedder,
        "structured_parser": cfg.pipelines.structured.parser,
        "formula_extractor": cfg.pipelines.structured.formula_extractor,
        "figure_descriptor": cfg.pipelines.structured.figure_descriptor,
        "retrieval_engine": cfg.retrieval.retrieval_engine,
        "fusion_engine": cfg.retrieval.fusion_engine,
    }


def _probe_dependencies(cfg: AppConfig) -> list[str]:
    issues: list[str] = []
    for role, adapter_name in _active_adapter_names(cfg).items():
        for module_name in _DEPENDENCY_MODULES.get(adapter_name, ()):
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - exercised via CLI tests
                issues.append(
                    f"{role}={adapter_name} failed to import {module_name}: "
                    f"{exc.__class__.__name__}: {exc}"
                )
    return issues


def _validate_index_layout(cfg: AppConfig, runtime: ActiveRuntime) -> list[str]:
    issues: list[str] = []
    runtime.index_writer.healthcheck()

    for key, collection_name in runtime.index_layout.collections.items():
        expected = runtime.index_layout.vector_schemas[key]
        actual = runtime.index_writer.get_collection_schema(collection_name)
        if actual is None:
            continue
        if not schema_matches(expected, actual):
            issues.append(
                f"{collection_name}: expected {expected.model_dump()} "
                f"but found {actual.model_dump()}"
            )

    if issues and cfg.retrieval.fail_on_schema_mismatch:
        return issues
    return issues


def _build_doc_meta(pdf_path: Path, renderer: Any) -> DocumentMeta:
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
    runtime = _build_runtime(cfg)

    if cfg.retrieval.validate_on_start:
        runtime.index_writer.healthcheck()

    doc_meta = _build_doc_meta(pdf_path, runtime.renderer)
    print(
        f"Ingesting {pdf_path.name} | doc_id={doc_meta.doc_id} | "
        f"pages={doc_meta.total_pages} | namespace="
        f"{runtime.index_layout.namespace or 'legacy'}"
    )

    visual_pipeline = VisualPipeline(
        renderer=runtime.renderer,
        embedder=runtime.visual_embedder,
        index_writer=runtime.index_writer,
        storage=storage,
        config=cfg.pipelines.visual,
        index_layout=runtime.index_layout,
        fail_on_schema_mismatch=cfg.retrieval.fail_on_schema_mismatch,
    )

    text_pipeline = TextPipeline(
        extractor=runtime.text_extractor,
        chunker=runtime.text_chunker,
        embedder=runtime.text_embedder,
        index_writer=runtime.index_writer,
        storage=storage,
        config=cfg.pipelines.text,
        index_layout=runtime.index_layout,
        fail_on_schema_mismatch=cfg.retrieval.fail_on_schema_mismatch,
    )

    structured_pipeline = StructuredPipeline(
        parser=runtime.structured_parser,
        formula_extractor=runtime.formula_extractor,
        figure_descriptor=runtime.figure_descriptor,
        embedder=runtime.text_embedder,
        index_writer=runtime.index_writer,
        storage=storage,
        config=cfg.pipelines.structured,
        renderer=runtime.renderer,
        index_layout=runtime.index_layout,
        fail_on_schema_mismatch=cfg.retrieval.fail_on_schema_mismatch,
    )

    pages = visual_pipeline.run(pdf_path, doc_meta)
    print(f"  Visual:     {len(pages)} pages indexed")

    chunks = text_pipeline.run(pdf_path, doc_meta)
    print(f"  Text:       {len(chunks)} chunks indexed")

    tables, formulas, figures = structured_pipeline.run(pdf_path, doc_meta)
    print(
        f"  Structured: {len(tables)} tables, {len(formulas)} formulas, "
        f"{len(figures)} figures indexed"
    )

    print(
        "  Collections: "
        f"visual={runtime.index_layout.collections['visual']}, "
        f"text={runtime.index_layout.collections['text']}, "
        f"tables={runtime.index_layout.collections['tables']}"
    )

    doc_dir = storage.doc_dir(doc_meta.doc_id)
    print(f"\n  Outputs: {doc_dir}")


def _run_doctor(cfg: AppConfig) -> None:
    issues = _probe_dependencies(cfg)

    try:
        runtime = _build_runtime(cfg)
    except Exception as exc:
        issues.append(
            f"runtime build failed: {exc.__class__.__name__}: {exc}\n"
            f"{traceback.format_exc(limit=5)}"
        )
        runtime = None

    if runtime is not None:
        try:
            issues.extend(_validate_index_layout(cfg, runtime))
        except Exception as exc:
            issues.append(
                f"backend validation failed: {exc.__class__.__name__}: {exc}\n"
                f"{traceback.format_exc(limit=5)}"
            )

    print("Doctor report")
    print(f"  Backend: {cfg.retrieval.retrieval_engine}")
    print(f"  Namespace mode: {cfg.retrieval.index_namespace}")
    if runtime is not None:
        print(
            f"  Resolved namespace: {runtime.index_layout.namespace or 'legacy'}"
        )
        print(
            "  Adapter signatures: "
            f"visual={runtime.index_layout.adapter_signatures['visual_embedder']}, "
            f"text={runtime.index_layout.adapter_signatures['text_embedder']}"
        )
        print("  Collections:")
        for key, value in runtime.index_layout.collections.items():
            print(f"    {key}: {value}")

    if issues:
        print("  Status: FAILED")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    print("  Status: OK")


def main() -> None:
    args = _parse_args()

    if args.command is None:
        print("Usage: python -m src [ingest|doctor] ...")
        sys.exit(1)

    cfg = _load_cfg(getattr(args, "config", None))

    if args.command == "doctor":
        _run_doctor(cfg)
        return

    if args.command == "ingest":
        _run_ingest(args.pdf, cfg)
        return

    print(f"Unknown command: {args.command}")
    sys.exit(1)


if __name__ == "__main__":
    main()
