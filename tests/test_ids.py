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
    def test_deterministic(self):
        id1 = generate_doc_id("/data/test.pdf")
        id2 = generate_doc_id("/data/test.pdf")
        assert id1 == id2

    def test_different_paths(self):
        id1 = generate_doc_id("/data/a.pdf")
        id2 = generate_doc_id("/data/b.pdf")
        assert id1 != id2

    def test_length(self):
        doc_id = generate_doc_id("/data/test.pdf")
        assert len(doc_id) == 16
