"""Stage functions for the extraction pipeline.

Each stage is a separate OS process invoked via the CLI. Stages share
marker semantics and a reporting helper so output is consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

StageName = Literal["segment", "extract-text", "describe-figures", "assemble"]

STAGE_ORDER: list[StageName] = [
    "segment",
    "extract-text",
    "describe-figures",
    "assemble",
]

OutcomeStatus = Literal["success", "skipped", "error", "missing_prereq"]


@dataclass
class StageOutcome:
    """Result for one (pdf|outdir) within a stage run."""

    label: str
    status: OutcomeStatus
    detail: str = ""


def next_stage(stage: StageName) -> StageName | None:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None


def print_stage_summary(
    stage: StageName,
    outcomes: list[StageOutcome],
    out_dirs_for_next: list[Path],
) -> int:
    """Print inline log already happened per path; this prints the end block.

    Returns the exit code: 0 if every outcome is success or skipped, else 1.
    """
    ok = sum(1 for o in outcomes if o.status in ("success", "skipped"))
    bad = sum(1 for o in outcomes if o.status in ("error", "missing_prereq"))
    bar = "━" * 44
    print()
    print(bar)
    if bad:
        print(f"  Stage '{stage}': {ok} erfolgreich, {bad} FEHLGESCHLAGEN")
    else:
        print(f"  Stage '{stage}': {ok} erfolgreich")
    print(bar)
    for o in outcomes:
        mark = {"success": "✓", "skipped": "↷", "error": "✗", "missing_prereq": "✗"}[o.status]
        suffix = f"  {o.detail}" if o.detail else ""
        print(f"  {mark} {o.label}{suffix}")
    print()
    nxt = next_stage(stage)
    if nxt is not None and out_dirs_for_next:
        paths = " ".join(str(p) for p in out_dirs_for_next)
        print("Nächster Schritt (nur erfolgreiche Ordner):")
        print(f"  python -m extraction {nxt} {paths}")
    elif nxt is None:
        print("Pipeline komplett.")
    print()
    return 0 if bad == 0 else 1
