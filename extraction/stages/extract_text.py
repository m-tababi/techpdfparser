"""Stage 2 — extract text content for TEXT/HEADING regions."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image as PILImage

from ..config import ExtractionConfig
from ..models import Element, ElementContent, ElementType, Region
from ..output import OutputWriter
from ..registry import get_text_extractor
from . import StageName, StageOutcome, print_stage_summary

_STAGE: StageName = "extract-text"
_PREV: StageName = "segment"
_TARGET_TYPES = {ElementType.TEXT, ElementType.HEADING}


def _element_id(doc_id: str, region: Region) -> str:
    x0, y0, x1, y1 = (round(v) for v in region.bbox)
    raw = f"{doc_id}:{region.page}:{region.region_type.value}:{x0},{y0},{x1},{y1}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_text(out_dirs: list[Path], cfg: ExtractionConfig) -> int:
    plan: list[tuple[Path, OutputWriter, dict]] = []
    outcomes: list[StageOutcome] = []

    for out_dir in out_dirs:
        writer = OutputWriter(out_dir)
        label = str(out_dir)
        if writer.is_stage_done(_STAGE):
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

    extractor = get_text_extractor(
        cfg.text_extractor, **cfg.get_adapter_config(cfg.text_extractor)
    )

    for out_dir, writer, meta in plan:
        label = str(out_dir)
        try:
            _process_one(out_dir, writer, meta, extractor, cfg)
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
    extractor: object,
    cfg: ExtractionConfig,
) -> None:
    regions: list[Region] = meta["regions"]
    doc_id: str = meta["doc_id"]
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
        if sidecar.exists():
            continue
        page_img = _load_page(out_dir, region.page)
        content: ElementContent = extractor.extract(page_img, region.page)  # type: ignore[attr-defined]
        if region.content is not None and region.content.caption:
            content.caption = region.content.caption
        if not (content.text or "").strip():
            continue
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
