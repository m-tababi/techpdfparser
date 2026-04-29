"""End-to-end integration against a real PDF using the default config.

Marker-gated: not run by `pytest -q` — the user triggers this manually on
a GPU machine with MinerU installed via `pytest -m integration -q`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PDF_REL = Path("1.9.20 PV 1001.12, Rev. 3.pdf")


@pytest.mark.integration
def test_default_pipeline_against_real_pdf(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = repo_root / PDF_REL
    if not pdf_path.exists():
        pytest.skip(f"test PDF missing: {pdf_path}")

    out_dir = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "extraction",
            "extract",
            str(pdf_path),
            "--output",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    content_list = out_dir / "content_list.json"
    assert content_list.exists()
    data = json.loads(content_list.read_text(encoding="utf-8"))
    assert data["total_pages"] > 0
    assert data["schema_version"] == "1.0"

    pages_dir = out_dir / "pages"
    assert pages_dir.exists()
    # At least page 0 must have a page.png.
    assert (pages_dir / "0" / "page.png").exists()

    # At least one sidecar somewhere under pages/.
    sidecars = list(pages_dir.rglob("*.json"))
    assert sidecars, "no per-element sidecars produced"

    # A technical spec like this one should yield at least one table.
    tables = [
        p
        for p in sidecars
        if p.name.endswith("_table.json")
    ]
    if tables:
        first_table = json.loads(tables[0].read_text(encoding="utf-8"))
        assert first_table["content"].get("markdown"), (
            "table sidecar missing markdown content"
        )
