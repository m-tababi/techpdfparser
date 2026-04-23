"""End-to-end integration test — all four stages on a real PDF.

Marked integration: requires GPU + real model weights. Run with:
    pytest -m integration extraction/tests/test_stages_integration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from extraction.config import ExtractionConfig
from extraction.stages.assemble import run_assemble
from extraction.stages.describe_figures import run_figures
from extraction.stages.extract_text import run_text
from extraction.stages.segment import run_segment

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).parent / "fixtures"
PDF_FIXTURE = REPO_ROOT / "3_HRA_for_offshore.pdf"
REFERENCE = FIXTURE_DIR / "reference_content_list.json"

STRICT_ELEMENT_KEYS = (
    "element_id", "type", "page", "bbox",
    "reading_order_index", "confidence", "extractor",
)
STRICT_CONTENT_KEYS = ("markdown", "latex", "caption", "image_path")
STRUCTURAL_CONTENT_KEYS = ("text", "description")


@pytest.mark.integration
def test_all_four_stages_bit_plus_structural(tmp_path: Path):
    assert PDF_FIXTURE.exists(), f"Fixture PDF missing: {PDF_FIXTURE}"
    assert REFERENCE.exists(), (
        f"Reference content_list.json missing at {REFERENCE}. "
        "Run the stages once on the fixture and copy the output there."
    )

    cfg = ExtractionConfig()  # GPU defaults: mineru25 / olmocr2 / mineru25 / noop / qwen25vl

    assert run_segment([PDF_FIXTURE], cfg, output_base=tmp_path) == 0
    out_dir = tmp_path / PDF_FIXTURE.stem
    assert run_text([out_dir], cfg) == 0
    assert run_figures([out_dir], cfg) == 0
    assert run_assemble([out_dir], cfg) == 0

    actual = json.loads((out_dir / "content_list.json").read_text(encoding="utf-8"))
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))

    for key in ("doc_id", "source_file", "total_pages", "schema_version",
                "segmentation_tool"):
        assert actual[key] == expected[key], f"top-level {key} drifted"

    assert actual["pages"] == expected["pages"]

    assert len(actual["elements"]) == len(expected["elements"]), (
        f"element count changed: {len(actual['elements'])} vs {len(expected['elements'])}"
    )
    for i, (a, e) in enumerate(zip(actual["elements"], expected["elements"])):
        for k in STRICT_ELEMENT_KEYS:
            assert a[k] == e[k], f"element[{i}].{k} drifted: {a[k]!r} vs {e[k]!r}"
        for k in STRICT_CONTENT_KEYS:
            assert a["content"].get(k) == e["content"].get(k), (
                f"element[{i}].content.{k} drifted"
            )
        for k in STRUCTURAL_CONTENT_KEYS:
            if e["content"].get(k):
                got = a["content"].get(k)
                assert isinstance(got, str) and got.strip(), (
                    f"element[{i}].content.{k} expected non-empty string"
                )
