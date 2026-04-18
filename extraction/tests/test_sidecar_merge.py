"""Sidecar → content_list merge must be deterministic.

Regardless of filesystem iteration order or input shuffle, build_content_list
must produce the same sorted output, and ties on (page, reading_order_index)
must be broken lexicographically by element_id.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from extraction.models import Element, ElementContent, ElementType
from extraction.output import OutputWriter


def _make(page: int, ro: int, el_id: str, kind: ElementType = ElementType.TEXT) -> Element:
    return Element(
        element_id=el_id,
        type=kind,
        page=page,
        bbox=[0, 0, 100, 50],
        reading_order_index=ro,
        confidence=0.9,
        extractor="test",
        content=ElementContent(text=f"{el_id}"),
    )


def test_build_content_list_is_deterministic_across_shuffles(tmp_path: Path) -> None:
    elements = [
        _make(0, 3, "1111111111111111"),
        _make(0, 1, "2222222222222222"),
        _make(1, 0, "3333333333333333"),
        _make(2, 2, "4444444444444444"),
        _make(2, 5, "5555555555555555"),
    ]

    rng = random.Random(42)
    for trial in range(5):
        trial_dir = tmp_path / f"trial_{trial}"
        writer = OutputWriter(trial_dir)
        shuffled = elements.copy()
        rng.shuffle(shuffled)
        for el in shuffled:
            writer.write_element_sidecar(el)

        cl = writer.build_content_list(
            doc_id="d", source_file="x.pdf", total_pages=3, segmentation_tool="mock"
        )
        writer.write_content_list(cl)

        dumped = json.loads((trial_dir / "content_list.json").read_text(encoding="utf-8"))
        dumped["source_file"] = "x.pdf"  # normalize across trials (equal anyway)
        if trial == 0:
            reference = dumped
        else:
            assert dumped == reference, f"trial {trial} diverged from trial 0"


def test_ties_broken_by_element_id(tmp_path: Path) -> None:
    # Same page, same reading_order_index → sort by element_id (lexicographic).
    elements = [
        _make(0, 5, "ffffffffffffffff"),
        _make(0, 5, "1111111111111111"),
        _make(0, 5, "aaaaaaaaaaaaaaaa"),
    ]
    writer = OutputWriter(tmp_path)
    for el in elements:
        writer.write_element_sidecar(el)

    cl = writer.build_content_list(
        doc_id="d", source_file="x.pdf", total_pages=1, segmentation_tool="mock"
    )

    assert [e.element_id for e in cl.elements] == [
        "1111111111111111",
        "aaaaaaaaaaaaaaaa",
        "ffffffffffffffff",
    ]
    # After tie-break, reading_order_index is re-numbered globally 0,1,2.
    assert [e.reading_order_index for e in cl.elements] == [0, 1, 2]
