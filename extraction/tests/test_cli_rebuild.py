"""CLI `rebuild` sub-command: derive content_list.json from sidecars alone."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from extraction.models import Element, ElementContent, ElementType
from extraction.output import OutputWriter


def _mk_element(page: int, ro: int, el_id: str, kind: ElementType) -> Element:
    return Element(
        element_id=el_id,
        type=kind,
        page=page,
        bbox=[0, 0, 100, 50],
        reading_order_index=ro,
        confidence=0.9,
        extractor="test",
        content=ElementContent(text=f"{el_id} on page {page}"),
    )


def test_rebuild_without_existing_content_list(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    writer.write_element_sidecar(_mk_element(0, 0, "a" * 16, ElementType.TEXT))
    writer.write_element_sidecar(_mk_element(1, 0, "b" * 16, ElementType.HEADING))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "extraction",
            "rebuild",
            str(tmp_path),
            "--doc-id",
            "doc1",
            "--source",
            "sample.pdf",
            "--pages",
            "2",
            "--segmenter",
            "mock",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "content_list.json").exists()
    data = json.loads((tmp_path / "content_list.json").read_text(encoding="utf-8"))
    assert data["doc_id"] == "doc1"
    assert data["total_pages"] == 2
    assert [e["element_id"] for e in data["elements"]] == ["a" * 16, "b" * 16]


def test_rebuild_uses_existing_content_list_metadata(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    writer.write_element_sidecar(_mk_element(0, 0, "c" * 16, ElementType.TEXT))

    # Seed a content_list.json so rebuild can pick up the metadata.
    (tmp_path / "content_list.json").write_text(
        json.dumps(
            {
                "doc_id": "seeded",
                "source_file": "seed.pdf",
                "total_pages": 1,
                "schema_version": "1.0",
                "segmentation_tool": "seed_seg",
                "pages": [],
                "elements": [],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "extraction", "rebuild", str(tmp_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "content_list.json").read_text(encoding="utf-8"))
    assert data["doc_id"] == "seeded"
    assert data["segmentation_tool"] == "seed_seg"
    assert len(data["elements"]) == 1
