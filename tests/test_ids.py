"""Test ID generation: stability, uniqueness, and collision properties."""

from src.utils.ids import generate_doc_id, generate_element_id


class TestGenerateElementId:
    def test_deterministic(self):
        id1 = generate_element_id("doc1", 0, "visual_page", "colqwen25")
        id2 = generate_element_id("doc1", 0, "visual_page", "colqwen25")
        assert id1 == id2

    def test_different_doc_ids(self):
        id1 = generate_element_id("doc1", 0, "visual_page", "colqwen25")
        id2 = generate_element_id("doc2", 0, "visual_page", "colqwen25")
        assert id1 != id2

    def test_different_pages(self):
        id1 = generate_element_id("doc1", 0, "visual_page", "colqwen25")
        id2 = generate_element_id("doc1", 1, "visual_page", "colqwen25")
        assert id1 != id2

    def test_different_tools(self):
        id1 = generate_element_id("doc1", 0, "visual_page", "colqwen25")
        id2 = generate_element_id("doc1", 0, "visual_page", "colpali")
        assert id1 != id2

    def test_different_types(self):
        id1 = generate_element_id("doc1", 0, "visual_page", "colqwen25")
        id2 = generate_element_id("doc1", 0, "text_chunk", "colqwen25")
        assert id1 != id2

    def test_sequence_differentiates(self):
        id1 = generate_element_id("doc1", 0, "table", "mineru25", 0)
        id2 = generate_element_id("doc1", 0, "table", "mineru25", 1)
        assert id1 != id2

    def test_length(self):
        # IDs are 16 hex chars
        eid = generate_element_id("doc1", 0, "formula", "ppformulanet")
        assert len(eid) == 16
        assert all(c in "0123456789abcdef" for c in eid)


class TestGenerateDocId:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 test content")
        id1 = generate_doc_id(str(f))
        id2 = generate_doc_id(str(f))
        assert id1 == id2

    def test_different_paths(self, tmp_path):
        # Same path, different content → different IDs
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"%PDF content A")
        b.write_bytes(b"%PDF content B")
        assert generate_doc_id(str(a)) != generate_doc_id(str(b))

    def test_same_content_different_paths(self, tmp_path):
        # Same content at two locations → same ID (content-stable)
        content = b"%PDF-1.4 identical"
        a = tmp_path / "a.pdf"
        b = tmp_path / "subdir" / "b.pdf"
        b.parent.mkdir()
        a.write_bytes(content)
        b.write_bytes(content)
        assert generate_doc_id(str(a)) == generate_doc_id(str(b))

    def test_length(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 test")
        doc_id = generate_doc_id(str(f))
        assert len(doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc_id)
