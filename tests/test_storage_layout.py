"""Tests for the document-centric StorageManager layout."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.storage import StorageManager


class TestPaths:
    def test_doc_dir_created(self, tmp_path):
        sm = StorageManager(tmp_path)
        d = sm.doc_dir("abc123")
        assert d.exists()
        assert d == tmp_path / "documents" / "abc123"

    def test_run_dir_inside_doc_dir(self, tmp_path):
        sm = StorageManager(tmp_path)
        rd = sm.run_dir("abc123", "visual", "clip")
        assert rd.exists()
        assert rd.parent.parent == tmp_path / "documents" / "abc123"
        assert rd.name.startswith("visual_clip_")

    def test_image_path_in_pages_subdir(self, tmp_path):
        sm = StorageManager(tmp_path)
        rd = sm.run_dir("abc123", "visual", "clip")
        p = sm.image_path(rd, 3)
        assert p.parent == rd / "pages"
        assert p.name == "p0003.png"

    def test_figure_path_in_figures_subdir(self, tmp_path):
        sm = StorageManager(tmp_path)
        rd = sm.run_dir("abc123", "structured", "pdfplumber")
        p = sm.figure_path(rd, 2, 1)
        assert p.parent == rd / "figures"
        assert p.name == "p0002_f1.png"

    def test_document_json_path(self, tmp_path):
        sm = StorageManager(tmp_path)
        p = sm.document_json_path("abc123")
        assert p == tmp_path / "documents" / "abc123" / "document.json"


class TestDocumentIndex:
    def test_creates_document_json_on_first_call(self, tmp_path):
        sm = StorageManager(tmp_path)
        sm.update_document_index("doc1", "test.pdf", "visual_clip_20260101", "visual")
        path = sm.document_json_path("doc1")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["doc_id"] == "doc1"
        assert len(data["runs"]) == 1
        assert data["runs"][0]["run_id"] == "visual_clip_20260101"

    def test_appends_new_run(self, tmp_path):
        sm = StorageManager(tmp_path)
        sm.update_document_index("doc1", "test.pdf", "visual_clip_20260101", "visual")
        sm.update_document_index("doc1", "test.pdf", "text_minilm_20260101", "text")
        data = json.loads(sm.document_json_path("doc1").read_text())
        assert len(data["runs"]) == 2

    def test_idempotent_on_same_run_id(self, tmp_path):
        sm = StorageManager(tmp_path)
        sm.update_document_index("doc1", "test.pdf", "visual_clip_20260101", "visual")
        sm.update_document_index("doc1", "test.pdf", "visual_clip_20260101", "visual")
        data = json.loads(sm.document_json_path("doc1").read_text())
        assert len(data["runs"]) == 1
