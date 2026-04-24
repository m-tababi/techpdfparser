from __future__ import annotations

from extraction.adapters.mineru25_segmenter import (
    _block_to_region,
    _confidence_for_block,
)
from extraction.models import ElementType
from extraction.registry import get_formula_extractor, get_text_extractor


def test_confidence_for_block_matches_by_bbox() -> None:
    layout_dets = [
        {"bbox": [10, 20, 100, 50], "score": 0.42},
        {"bbox": [10, 60, 100, 200], "score": 0.91},
    ]
    block = {"bbox": [10, 60, 100, 200], "type": "text"}
    assert _confidence_for_block(block, layout_dets) == 0.91


def test_confidence_for_block_defaults_to_one_when_missing() -> None:
    layout_dets = [{"bbox": [0, 0, 1, 1], "score": 0.5}]
    block = {"bbox": [10, 20, 30, 40], "type": "text"}
    assert _confidence_for_block(block, layout_dets) == 1.0


def test_block_to_region_uses_layout_dets_confidence() -> None:
    layout_dets = [{"bbox": [0, 0, 100, 50], "score": 0.33}]
    block = {
        "bbox": [0, 0, 100, 50],
        "type": "text",
        "lines": [{"spans": [{"type": "text", "content": "hello"}]}],
    }
    region = _block_to_region(block, page_number=3, layout_dets=layout_dets)
    assert region is not None
    assert region.region_type == ElementType.TEXT
    assert abs(region.confidence - 0.33) < 1e-6


def test_mineru25_text_and_formula_passthroughs_are_registered() -> None:
    text = get_text_extractor("mineru25")
    formula = get_formula_extractor("mineru25")
    assert text.tool_name == "mineru25"
    assert formula.tool_name == "mineru25"
