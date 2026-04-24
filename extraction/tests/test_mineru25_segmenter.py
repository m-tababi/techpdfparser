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


def test_confidence_for_block_prefers_direct_block_score() -> None:
    # MinerU 2.5 writes the score directly on each para_block; layout_dets is
    # empty. Prefer the direct score.
    block = {"bbox": [10, 60, 100, 200], "type": "text", "score": 0.87}
    assert _confidence_for_block(block, layout_dets=[]) == 0.87


def test_confidence_for_block_uses_iou_match_when_no_direct_score() -> None:
    # Fallback path: no direct score → find best-overlapping layout_det.
    # Block [0,0,120,120] vs det [10,10,110,110] → IoU = 10000/14400 ≈ 0.69.
    layout_dets = [
        {"bbox": [10, 10, 110, 110], "score": 0.75},
        {"bbox": [500, 500, 600, 600], "score": 0.99},
    ]
    block = {"bbox": [0, 0, 120, 120], "type": "text"}
    assert _confidence_for_block(block, layout_dets) == 0.75


def test_confidence_for_block_picks_best_iou_not_highest_score() -> None:
    # Two dets overlap: one with high IoU and moderate score, one with low
    # IoU and high score. Should return the high-IoU det's score.
    layout_dets = [
        {"bbox": [0, 0, 100, 100], "score": 0.60},    # IoU with block = 1.0
        {"bbox": [0, 0, 200, 200], "score": 0.99},    # IoU = 100*100/40000 = 0.25
    ]
    block = {"bbox": [0, 0, 100, 100], "type": "text"}
    assert _confidence_for_block(block, layout_dets) == 0.60


def test_table_block_keeps_raw_html_with_rowspan_colspan() -> None:
    # Hierarchical header: 'Group' spans two sub-columns; markdown flattens
    # this, html must preserve it.
    html = (
        "<table><tr><td rowspan=\"2\">Mat</td>"
        "<td colspan=\"2\">Mech</td></tr>"
        "<tr><td>0</td><td>90</td></tr>"
        "<tr><td>steel</td><td>180</td><td>204</td></tr></table>"
    )
    block = {
        "bbox": [0.0, 0.0, 100.0, 100.0],
        "type": "table",
        "score": 0.95,
        "blocks": [
            {
                "type": "table_body",
                "lines": [{"spans": [{"type": "table", "html": html, "image_path": ""}]}],
            }
        ],
    }
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.region_type == ElementType.TABLE
    assert region.content is not None
    assert region.content.html == html
    assert 'rowspan="2"' in region.content.html
    assert 'colspan="2"' in region.content.html
    # markdown stays as the flat fallback
    assert region.content.markdown is not None
    assert "Mat" in region.content.markdown
