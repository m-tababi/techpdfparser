"""Stage 1 — segment PDFs and write passthrough sidecars."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import ExtractionConfig
from ..models import Element, ElementType, Region
from ..output import OutputWriter
from ..registry import get_renderer, get_segmenter
from . import StageName, StageOutcome, print_stage_summary

_STAGE: StageName = "segment"


def _doc_id(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _element_id(doc_id: str, region: Region) -> str:
    x0, y0, x1, y1 = (round(v) for v in region.bbox)
    raw = f"{doc_id}:{region.page}:{region.region_type.value}:{x0},{y0},{x1},{y1}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _role_tool_name(region_type: ElementType, cfg: ExtractionConfig) -> str:
    if region_type in (ElementType.TEXT, ElementType.HEADING):
        return cfg.text_extractor
    if region_type == ElementType.TABLE:
        return cfg.table_extractor
    if region_type == ElementType.FORMULA:
        return cfg.formula_extractor
    return cfg.figure_descriptor


def run_segment(
    pdf_paths: list[Path],
    cfg: ExtractionConfig,
    output_base: Path,
) -> int:
    plan: list[tuple[Path, Path, OutputWriter]] = []
    outcomes: list[StageOutcome] = []
    for pdf in pdf_paths:
        out_dir = output_base / pdf.stem
        pre_existed = out_dir.exists() and any(out_dir.iterdir())
        writer = OutputWriter(out_dir)
        label = str(out_dir)
        if writer.is_stage_done(_STAGE):
            outcomes.append(StageOutcome(label=label, status="skipped"))
            print(f"Processing {label} ... ↷ skipped (already done)")
            continue
        if pre_existed:
            exc = FileExistsError(
                f"Output directory '{out_dir}' already contains artefacts but "
                f"'.stages/{_STAGE}.done' is missing. Delete the directory or "
                f"its orphaned contents before re-running segment."
            )
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="error",
                detail=f"(Ordner nicht leer, kein segment.done; siehe .stages/{_STAGE}.error)",
            ))
            print(f"Processing {label} ... ✗ dirty output directory without segment.done")
            continue
        plan.append((pdf, out_dir, writer))

    if not plan:
        ok_out_dirs = [output_base / p.stem for p in pdf_paths]
        return print_stage_summary(_STAGE, outcomes, ok_out_dirs)

    renderer_kwargs = dict(cfg.get_adapter_config(cfg.renderer))
    renderer_kwargs["dpi"] = cfg.resolve_renderer_dpi()
    renderer = get_renderer(cfg.renderer, **renderer_kwargs)
    segmenter = get_segmenter(cfg.segmenter, **cfg.get_adapter_config(cfg.segmenter))

    for pdf, out_dir, writer in plan:
        label = str(out_dir)
        try:
            _process_one(pdf, writer, renderer, segmenter, cfg)
            writer.mark_stage_done(_STAGE)
            seg = writer.read_segmentation()
            outcomes.append(StageOutcome(
                label=label, status="success",
                detail=f"({seg['total_pages']} Seiten, {len(seg['regions'])} Regions)",
            ))
            print(f"Processing {label} ... ✓")
        except Exception as exc:
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="error",
                detail=f"(Fehler: siehe .stages/{_STAGE}.error)",
            ))
            print(f"Processing {label} ... ✗ {type(exc).__name__}: {exc}")

    ok_out_dirs = [
        output_base / p.stem for p in pdf_paths
        if (output_base / p.stem / ".stages" / f"{_STAGE}.done").exists()
    ]
    return print_stage_summary(_STAGE, outcomes, ok_out_dirs)


def _process_one(
    pdf_path: Path,
    writer: OutputWriter,
    renderer: object,
    segmenter: object,
    cfg: ExtractionConfig,
) -> None:
    page_count = renderer.page_count(pdf_path)  # type: ignore[attr-defined]
    page_images = []
    for i in range(page_count):
        img = renderer.render_page(pdf_path, i)  # type: ignore[attr-defined]
        writer.save_page_image(page=i, image=img)
        page_images.append(img)

    regions = segmenter.segment(pdf_path)  # type: ignore[attr-defined]

    doc_id = _doc_id(pdf_path)
    writer.write_segmentation(
        regions=regions,
        doc_id=doc_id,
        source_file=pdf_path.name,
        total_pages=page_count,
        segmentation_tool=segmenter.tool_name,  # type: ignore[attr-defined]
    )

    seg_tool = segmenter.tool_name  # type: ignore[attr-defined]
    for region in regions:
        if _role_tool_name(region.region_type, cfg) != seg_tool:
            continue
        if region.content is None:
            continue
        if region.confidence < cfg.confidence_threshold:
            continue
        el_id = _element_id(doc_id, region)
        el = Element(
            element_id=el_id,
            type=region.region_type,
            page=region.page,
            bbox=region.bbox,
            reading_order_index=region.reading_order_index,
            section_path=[],
            confidence=region.confidence,
            extractor=seg_tool,
            content=region.content.model_copy(),
        )
        if region.region_type in {
            ElementType.TABLE, ElementType.FORMULA, ElementType.FIGURE,
            ElementType.DIAGRAM, ElementType.TECHNICAL_DRAWING,
        } and 0 <= region.page < len(page_images):
            crop = writer.crop_region(page_images[region.page], region.bbox, dpi=cfg.resolve_renderer_dpi())
            rel = writer.save_element_crop(
                page=region.page, element_id=el_id,
                element_type=region.region_type.value, image=crop,
            )
            el.content.image_path = str(rel.relative_to(writer.output_dir))
        writer.write_element_sidecar(el)
