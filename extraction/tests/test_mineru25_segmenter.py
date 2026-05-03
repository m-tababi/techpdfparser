from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from extraction.adapters.mineru25_segmenter import (
    MinerU25Segmenter,
    MinerUHybridSegmenter,
    MinerUVLMSegmenter,
    _block_to_region,
    _confidence_for_block,
)
from extraction.models import CellMarker, ElementType, TableFootnote
from extraction.registry import (
    get_formula_extractor,
    get_segmenter,
    get_table_extractor,
    get_text_extractor,
)


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


@pytest.mark.parametrize(
    "adapter_name",
    ["mineru25", "mineru_hybrid", "mineru_vlm"],
)
def test_mineru_segmenters_are_registered(adapter_name: str) -> None:
    assert get_segmenter(adapter_name).tool_name == adapter_name


@pytest.mark.parametrize(
    "adapter_name",
    ["mineru25", "mineru_hybrid", "mineru_vlm"],
)
def test_mineru_passthrough_roles_are_registered(adapter_name: str) -> None:
    assert get_text_extractor(adapter_name).tool_name == adapter_name
    assert get_table_extractor(adapter_name).tool_name == adapter_name
    assert get_formula_extractor(adapter_name).tool_name == adapter_name


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


def _table_block_with_caption(
    table_bbox: list[float],
    caption_bboxes: list[list[float]],
    caption_text: str = "Table 1.",
) -> dict[str, Any]:
    body_block = {
        "type": "table_body",
        "bbox": list(table_bbox),
        "lines": [
            {"spans": [{"type": "table", "html": "<table><tr><td>x</td></tr></table>"}]}
        ],
    }
    caption_blocks = [
        {
            "type": "table_caption",
            "bbox": list(cap_bbox),
            "lines": [{"spans": [{"type": "text", "content": caption_text}]}],
        }
        for cap_bbox in caption_bboxes
    ]
    return {
        "bbox": list(table_bbox),
        "type": "table",
        "score": 0.9,
        "blocks": [body_block, *caption_blocks],
    }


def test_table_caption_above_sets_caption_position_above() -> None:
    block = _table_block_with_caption(
        table_bbox=[50.0, 200.0, 300.0, 400.0],
        caption_bboxes=[[50.0, 100.0, 300.0, 180.0]],
    )
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.caption_position == "above"


def test_table_caption_below_sets_caption_position_below() -> None:
    block = _table_block_with_caption(
        table_bbox=[50.0, 200.0, 300.0, 400.0],
        caption_bboxes=[[50.0, 410.0, 300.0, 440.0]],
    )
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.caption_position == "below"


def test_table_caption_overlapping_sets_caption_position_none() -> None:
    block = _table_block_with_caption(
        table_bbox=[50.0, 200.0, 300.0, 400.0],
        caption_bboxes=[[50.0, 380.0, 300.0, 420.0]],
    )
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.caption_position is None


def test_table_without_caption_block_keeps_caption_position_none() -> None:
    block = {
        "bbox": [0.0, 0.0, 100.0, 100.0],
        "type": "table",
        "score": 0.95,
        "blocks": [
            {
                "type": "table_body",
                "bbox": [0.0, 0.0, 100.0, 100.0],
                "lines": [
                    {"spans": [{"type": "table", "html": "<table><tr><td>x</td></tr></table>"}]}
                ],
            }
        ],
    }
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.caption_position is None


def test_table_with_multiple_captions_all_above_sets_caption_position_above() -> None:
    block = _table_block_with_caption(
        table_bbox=[50.0, 200.0, 300.0, 400.0],
        caption_bboxes=[
            [50.0, 100.0, 300.0, 130.0],
            [50.0, 140.0, 300.0, 180.0],
        ],
    )
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.caption_position == "above"


def _table_block_with_footnotes(footnote_texts: list[str]) -> dict[str, Any]:
    body_block = {
        "type": "table_body",
        "bbox": [50.0, 200.0, 300.0, 400.0],
        "lines": [
            {"spans": [{"type": "table", "html": "<table><tr><td>x</td></tr></table>"}]}
        ],
    }
    footnote_blocks = [
        {
            "type": "table_footnote",
            "bbox": [50.0, 410.0 + 20.0 * i, 300.0, 425.0 + 20.0 * i],
            "lines": [{"spans": [{"type": "text", "content": text}]}],
        }
        for i, text in enumerate(footnote_texts)
    ]
    return {
        "bbox": [50.0, 200.0, 300.0, 500.0],
        "type": "table",
        "score": 0.9,
        "blocks": [body_block, *footnote_blocks],
    }


def test_table_with_single_footnote_sets_footnotes_list() -> None:
    block = _table_block_with_footnotes(["* Values are in mg/L."])
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.footnotes == [TableFootnote(text="* Values are in mg/L.")]


def test_table_with_multiple_footnotes_preserves_order() -> None:
    block = _table_block_with_footnotes([
        "a) First note",
        "b) Second note",
    ])
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.footnotes == [
        TableFootnote(text="a) First note"),
        TableFootnote(text="b) Second note"),
    ]


def test_table_without_footnote_block_keeps_footnotes_none() -> None:
    block = {
        "bbox": [0.0, 0.0, 100.0, 100.0],
        "type": "table",
        "score": 0.95,
        "blocks": [
            {
                "type": "table_body",
                "bbox": [0.0, 0.0, 100.0, 100.0],
                "lines": [
                    {"spans": [{"type": "table", "html": "<table><tr><td>x</td></tr></table>"}]}
                ],
            }
        ],
    }
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.footnotes is None


def test_table_with_empty_footnote_text_drops_field() -> None:
    block = _table_block_with_footnotes(["   "])
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.footnotes is None


def _table_block_with_html(html: str) -> dict[str, Any]:
    return {
        "bbox": [0.0, 0.0, 100.0, 100.0],
        "type": "table",
        "score": 0.9,
        "blocks": [
            {
                "type": "table_body",
                "bbox": [0.0, 0.0, 100.0, 100.0],
                "lines": [{"spans": [{"type": "table", "html": html, "image_path": ""}]}],
            }
        ],
    }


def test_table_with_numeric_sup_marker_emits_cell_marker() -> None:
    html = "<table><tr><td>4.2<sup>a</sup></td></tr></table>"
    region = _block_to_region(_table_block_with_html(html), page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.markers == [CellMarker(value="4.2", marker="a")]


def test_table_marker_skips_cell_with_non_numeric_text_before_sup() -> None:
    html = "<table><tr><td>Total<sup>a</sup></td></tr></table>"
    region = _block_to_region(_table_block_with_html(html), page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.markers is None


def test_table_marker_skips_mixed_alphanumeric_text_before_sup() -> None:
    html = "<table><tr><td>n=12 <sup>a</sup></td></tr></table>"
    region = _block_to_region(_table_block_with_html(html), page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.markers is None


def test_table_with_no_sup_anywhere_keeps_markers_none() -> None:
    html = "<table><tr><td>4.2</td><td>1.5</td></tr></table>"
    region = _block_to_region(_table_block_with_html(html), page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.markers is None


def test_table_with_multiple_marker_cells_preserves_order() -> None:
    html = (
        "<table>"
        "<tr><td>0.236<sup>+</sup></td><td>0.924<sup>+</sup></td></tr>"
        "<tr><td>0.044<sup>+</sup></td></tr>"
        "</table>"
    )
    region = _block_to_region(_table_block_with_html(html), page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.markers == [
        CellMarker(value="0.236", marker="+"),
        CellMarker(value="0.924", marker="+"),
        CellMarker(value="0.044", marker="+"),
    ]


def test_table_marker_handles_leading_whitespace_before_value() -> None:
    # Pretty-printed HTML with newline / spaces before the numeric text
    # should still emit the marker — _cell_markers_from_html strips() the
    # leading text node before applying the numeric regex.
    html = "<table><tr><td>\n   4.2<sup>a</sup></td></tr></table>"
    region = _block_to_region(_table_block_with_html(html), page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.markers == [CellMarker(value="4.2", marker="a")]


def test_table_marker_takes_first_sup_when_cell_has_multiple_sups() -> None:
    html = "<table><tr><td>4.2<sup>a</sup><sup>b</sup></td></tr></table>"
    region = _block_to_region(_table_block_with_html(html), page_number=0, layout_dets=[])
    assert region is not None
    assert region.content is not None
    assert region.content.markers == [CellMarker(value="4.2", marker="a")]


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


@pytest.mark.parametrize(
    ("adapter_cls", "expected_backend"),
    [
        (MinerU25Segmenter, "pipeline"),
        (MinerUHybridSegmenter, "hybrid-auto-engine"),
        (MinerUVLMSegmenter, "vlm-auto-engine"),
    ],
)
def test_mineru_segmenters_pass_expected_backend(
    tmp_path: Path,
    adapter_cls: type[Any],
    expected_backend: str,
) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    seen_backend: list[str] = []

    def _fake_do_parse(**kwargs: object) -> None:
        seen_backend.append(str(kwargs["backend"]))
        output_dir = Path(str(kwargs["output_dir"]))
        middle_dir = output_dir / "sample.pdf" / "auto"
        middle_dir.mkdir(parents=True)
        (middle_dir / "sample_middle.json").write_text(
            json.dumps({"pdf_info": []}),
            encoding="utf-8",
        )

    adapter = adapter_cls()
    adapter._do_parse = _fake_do_parse

    assert adapter.segment(pdf) == []
    assert seen_backend == [expected_backend]


@pytest.mark.parametrize("block_type", ["list", "code", "algorithm"])
def test_vlm_text_like_blocks_map_to_text(block_type: str) -> None:
    block = {
        "bbox": [0, 0, 100, 50],
        "type": block_type,
        "content": "step one\nstep two",
        "score": 0.8,
    }
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.region_type == ElementType.TEXT
    assert region.content is not None
    assert region.content.text == "step one\nstep two"


def test_equation_block_maps_to_formula_from_direct_content() -> None:
    block = {
        "bbox": [0, 0, 100, 50],
        "type": "equation",
        "content": "E = mc^2",
        "score": 0.9,
    }
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.region_type == ElementType.FORMULA
    assert region.content is not None
    assert region.content.latex == "E = mc^2"


def test_chart_block_maps_to_diagram_with_direct_caption() -> None:
    block = {
        "bbox": [0, 0, 100, 50],
        "type": "chart",
        "chart_caption": "Figure 1. Stress curve",
        "score": 0.9,
    }
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.region_type == ElementType.DIAGRAM
    assert region.content is not None
    assert region.content.caption == "Figure 1. Stress curve"


def test_text_block_reads_lines_and_untyped_spans() -> None:
    block = {
        "bbox": [0, 0, 100, 50],
        "type": "text",
        "lines": [{"spans": [{"content": "plain span"}]}],
        "score": 0.9,
    }
    region = _block_to_region(block, page_number=0, layout_dets=[])
    assert region is not None
    assert region.region_type == ElementType.TEXT
    assert region.content is not None
    assert region.content.text == "plain span"


def test_mineru_segmenter_cleans_temporary_parse_dir(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    seen_output_dirs: list[Path] = []

    def _fake_do_parse(**kwargs: object) -> None:
        output_dir = Path(str(kwargs["output_dir"]))
        seen_output_dirs.append(output_dir)
        middle_dir = output_dir / "sample.pdf" / "auto"
        middle_dir.mkdir(parents=True)
        (middle_dir / "sample_middle.json").write_text(
            json.dumps({"pdf_info": []}),
            encoding="utf-8",
        )

    adapter = MinerU25Segmenter()
    adapter._do_parse = _fake_do_parse

    assert adapter.segment(pdf) == []
    assert seen_output_dirs
    assert not seen_output_dirs[0].exists()
