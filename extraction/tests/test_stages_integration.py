"""End-to-end integration test — all four stages as real CLI subprocesses.

Marked integration: requires GPU + real model weights. Run with:
    pytest -m integration extraction/tests/test_stages_integration.py

Why subprocess, not in-process: each stage is meant to run in its own OS
process so the kernel releases GPU memory between stages. Invoking the
four run_* functions in one pytest process would keep MinerU + olmOCR-2
+ Qwen2.5-VL resident simultaneously and OOM on anything short of a ~40
GB GPU. The test mirrors the real usage — `python -m extraction <stage>`
per stage — which is both more honest and physically runnable on the
24 GB hardware the OOM fix targets.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def _run_stage(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "extraction", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Stage {args} failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


@pytest.mark.integration
def test_all_four_stages_bit_plus_structural(tmp_path: Path) -> None:
    assert PDF_FIXTURE.exists(), f"Fixture PDF missing: {PDF_FIXTURE}"
    assert REFERENCE.exists(), (
        f"Reference content_list.json missing at {REFERENCE}. "
        "Run the four CLI stages once on the fixture and copy the "
        "resulting content_list.json there."
    )

    _run_stage(["segment", str(PDF_FIXTURE), "--out", str(tmp_path)])
    out_dir = tmp_path / PDF_FIXTURE.stem
    _run_stage(["extract-text", str(out_dir)])
    _run_stage(["describe-figures", str(out_dir)])
    _run_stage(["assemble", str(out_dir)])

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
