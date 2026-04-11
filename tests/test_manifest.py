"""Tests for ManifestBuilder."""
from __future__ import annotations

import json

from src.utils.manifest import ManifestBuilder


def _make_manifest(**kwargs) -> ManifestBuilder:
    defaults = dict(
        run_id="visual_clip_20260101_120000",
        pipeline="visual",
        doc_id="doc1",
        source_file="test.pdf",
        tools={"embedder": "clip"},
    )
    defaults.update(kwargs)
    return ManifestBuilder(**defaults)


class TestManifestBuilder:
    def test_write_creates_manifest_json(self, tmp_path):
        m = _make_manifest()
        m.write(tmp_path)
        assert (tmp_path / "manifest.json").exists()

    def test_fields_present(self, tmp_path):
        m = _make_manifest()
        m.set_counts(pages=9)
        m.set_qdrant_info("visual_pages", 9)
        m.write(tmp_path)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["run_id"] == "visual_clip_20260101_120000"
        assert data["pipeline"] == "visual"
        assert data["counts"]["pages"] == 9
        assert data["qdrant"]["upserted"] == 9

    def test_duration_is_non_negative(self, tmp_path):
        m = _make_manifest()
        m.write(tmp_path)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["duration_seconds"] >= 0

    def test_tool_versions_recorded(self, tmp_path):
        m = _make_manifest()
        m.set_tool_version("clip", "4.40")
        m.write(tmp_path)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["tool_versions"]["clip"] == "4.40"

    def test_empty_optional_fields_dont_crash(self, tmp_path):
        m = _make_manifest()
        # No counts, no qdrant info set — should still write cleanly
        m.write(tmp_path)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["counts"] == {}
        assert data["qdrant"] == {}
