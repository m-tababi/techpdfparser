"""Stage 2 — extract text content for TEXT/HEADING regions."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image as PILImage

from ..config import ExtractionConfig
from ..models import Element, ElementContent, ElementType, Region
from ..output import OutputWriter
from ..registry import get_formula_extractor, get_table_extractor, get_text_extractor
from . import StageName, StageOutcome, print_stage_summary

_STAGE: StageName = "extract-text"
_PREV: StageName = "segment"
_TARGET_TYPES = {
    ElementType.TEXT, ElementType.HEADING,
    ElementType.TABLE, ElementType.FORMULA,
}


def _element_id(doc_id: str, region: Region) -> str:
    x0, y0, x1, y1 = (round(v) for v in region.bbox)
    raw = f"{doc_id}:{region.page}:{region.region_type.value}:{x0},{y0},{x1},{y1}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_text(out_dirs: list[Path], cfg: ExtractionConfig, *, force: bool = False) -> int:
    plan: list[tuple[Path, OutputWriter, dict]] = []
    outcomes: list[StageOutcome] = []

    for out_dir in out_dirs:
        writer = OutputWriter(out_dir)
        label = str(out_dir)
        if writer.is_stage_done(_STAGE) and not force:
            outcomes.append(StageOutcome(label=label, status="skipped"))
            print(f"Processing {label} ... ↷ skipped (already done)")
            continue
        if not writer.is_stage_done(_PREV):
            exc = FileNotFoundError(f"Stage '{_PREV}' not done for {out_dir}")
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="missing_prereq",
                detail=f"(Vorgänger '{_PREV}' fehlt)",
            ))
            print(f"Processing {label} ... ✗ missing prerequisite: {_PREV}")
            continue
        meta = writer.read_segmentation()
        plan.append((out_dir, writer, meta))

    if not plan:
        return print_stage_summary(_STAGE, outcomes, [
            d for d in out_dirs
            if (d / ".stages" / f"{_STAGE}.done").exists()
        ])

    text_extractor = get_text_extractor(
        cfg.text_extractor, **cfg.get_adapter_config(cfg.text_extractor)
    )
    table_extractor = get_table_extractor(
        cfg.table_extractor, **cfg.get_adapter_config(cfg.table_extractor)
    )
    formula_extractor = get_formula_extractor(
        cfg.formula_extractor, **cfg.get_adapter_config(cfg.formula_extractor)
    )
    extractors: dict[ElementType, object] = {
        ElementType.TEXT: text_extractor,
        ElementType.HEADING: text_extractor,
        ElementType.TABLE: table_extractor,
        ElementType.FORMULA: formula_extractor,
    }

    for out_dir, writer, meta in plan:
        label = str(out_dir)
        try:
            _process_one(out_dir, writer, meta, extractors, cfg, force=force)
            writer.clear_stage_done("assemble")
            writer.mark_stage_done(_STAGE)
            outcomes.append(StageOutcome(label=label, status="success"))
            print(f"Processing {label} ... ✓")
        except Exception as exc:
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="error",
                detail=f"(Fehler: siehe .stages/{_STAGE}.error)",
            ))
            print(f"Processing {label} ... ✗ {type(exc).__name__}: {exc}")

    ok_dirs = [
        d for d in out_dirs
        if (d / ".stages" / f"{_STAGE}.done").exists()
    ]
    return print_stage_summary(_STAGE, outcomes, ok_dirs)


def _load_page(out_dir: Path, page: int) -> PILImage.Image:
    path = out_dir / "pages" / str(page) / "page.png"
    return PILImage.open(path).convert("RGB")


def _process_one(
    out_dir: Path,
    writer: OutputWriter,
    meta: dict,
    extractors: dict[ElementType, object],
    cfg: ExtractionConfig,
    *,
    force: bool = False,
) -> None:
    regions: list[Region] = meta["regions"]
    doc_id: str = meta["doc_id"]
    render_dpi = int(meta.get("render_dpi") or cfg.resolve_renderer_dpi())
    for region in regions:
        if region.region_type not in _TARGET_TYPES:
            continue
        if region.confidence < cfg.confidence_threshold:
            continue
        el_id = _element_id(doc_id, region)
        sidecar = (
            out_dir / "pages" / str(region.page)
            / f"{el_id}_{region.region_type.value}.json"
        )
        crop_path = (
            out_dir / "pages" / str(region.page)
            / f"{el_id}_{region.region_type.value}.png"
        )
        if sidecar.exists() and not force:
            continue
        if force:
            if sidecar.exists():
                sidecar.unlink()
            if crop_path.exists():
                crop_path.unlink()
        extractor = extractors[region.region_type]
        page_img = _load_page(out_dir, region.page)
        crop = writer.crop_region(
            page_img, region.bbox, dpi=render_dpi
        )
        content: ElementContent = extractor.extract(crop, region.page)  # type: ignore[attr-defined]
        if region.content is not None and region.content.caption:
            content.caption = region.content.caption
        if region.region_type in (ElementType.TEXT, ElementType.HEADING):
            if not (content.text or "").strip():
                continue
        else:
            # TABLE / FORMULA: Crop + image_path persistieren,
            # auch wenn markdown/latex/text leer sind.
            rel = writer.save_element_crop(
                page=region.page, element_id=el_id,
                element_type=region.region_type.value, image=crop,
            )
            content.image_path = str(rel.relative_to(writer.output_dir))
        el = Element(
            element_id=el_id,
            type=region.region_type,
            page=region.page,
            bbox=region.bbox,
            reading_order_index=region.reading_order_index,
            section_path=[],
            confidence=region.confidence,
            extractor=extractor.tool_name,  # type: ignore[attr-defined]
            content=content,
        )
        writer.write_element_sidecar(el)
