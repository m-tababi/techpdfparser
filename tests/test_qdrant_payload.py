"""Tests for Qdrant payload serialization helpers."""

from src.adapters.vectordb.qdrant import _base_payload, _payload_to_element
from src.core.models.document import BoundingBox
from src.core.models.elements import Formula, TextChunk


def test_text_chunk_payload_roundtrip_preserves_identity_and_context():
    chunk = TextChunk(
        object_id="abc123def4567890",
        doc_id="doc1",
        source_file="test.pdf",
        page_number=2,
        tool_name="mock",
        tool_version="1.0",
        content="hello world",
        char_start=5,
        char_end=16,
        bbox=BoundingBox(x0=1, y0=2, x1=3, y1=4),
        raw_output_path="/tmp/chunk.json",
        parent_id="parent-1",
        child_ids=["child-1"],
    )
    chunk.section_title = "Methods"
    chunk.section_path = ["Methods", "2.1 Setup"]
    chunk.heading_level = 2

    payload = {
        **_base_payload(chunk),
        "content": chunk.content,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }
    restored = _payload_to_element(payload)

    assert restored.object_id == chunk.object_id
    assert restored.bbox == chunk.bbox
    assert restored.char_start == chunk.char_start
    assert restored.char_end == chunk.char_end
    assert restored.raw_output_path == chunk.raw_output_path
    assert restored.parent_id == chunk.parent_id
    assert restored.child_ids == chunk.child_ids
    assert restored.section_title == "Methods"
    assert restored.section_path == ["Methods", "2.1 Setup"]
    assert restored.heading_level == 2


def test_formula_payload_roundtrip_preserves_optional_image_path():
    formula = Formula(
        object_id="fff123def4567890",
        doc_id="doc1",
        source_file="test.pdf",
        page_number=0,
        tool_name="ppformulanet",
        tool_version="1.0",
        latex=r"E = mc^2",
        content=r"E = mc^2",
        image_path="/tmp/formula.png",
    )

    payload = {
        **_base_payload(formula),
        "latex": formula.latex,
        "content": formula.content,
        "image_path": formula.image_path,
    }
    restored = _payload_to_element(payload)

    assert restored.object_id == formula.object_id
    assert restored.image_path == formula.image_path
