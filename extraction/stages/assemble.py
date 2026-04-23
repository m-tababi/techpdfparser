"""Stage 4 — assemble content_list.json from sidecars (no GPU)."""
from __future__ import annotations

from pathlib import Path

from ..config import ExtractionConfig
from ..output import OutputWriter
from . import StageName, StageOutcome, print_stage_summary

_STAGE: StageName = "assemble"
_PREREQS: tuple[StageName, ...] = ("segment", "extract-text", "describe-figures")


def run_assemble(out_dirs: list[Path], cfg: ExtractionConfig) -> int:  # noqa: ARG001
    outcomes: list[StageOutcome] = []
    for out_dir in out_dirs:
        writer = OutputWriter(out_dir)
        label = str(out_dir)
        if writer.is_stage_done(_STAGE):
            outcomes.append(StageOutcome(label=label, status="skipped"))
            print(f"Processing {label} ... ↷ skipped (already done)")
            continue
        missing = [p for p in _PREREQS if not writer.is_stage_done(p)]
        if missing:
            exc = FileNotFoundError(
                f"Stages {missing} not done for {out_dir}"
            )
            writer.write_stage_error(_STAGE, exc)
            outcomes.append(StageOutcome(
                label=label, status="missing_prereq",
                detail=f"(Vorgänger fehlt: {', '.join(missing)})",
            ))
            print(f"Processing {label} ... ✗ missing prerequisites: {missing}")
            continue
        try:
            _process_one(writer)
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


def _process_one(writer: OutputWriter) -> None:
    meta = writer.read_segmentation()
    content_list = writer.build_content_list(
        doc_id=meta["doc_id"],
        source_file=meta["source_file"],
        total_pages=meta["total_pages"],
        segmentation_tool=meta["segmentation_tool"],
    )
    writer.write_content_list(content_list)
