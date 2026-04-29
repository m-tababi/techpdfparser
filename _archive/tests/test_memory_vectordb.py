"""Tests for the in-memory vector DB backend."""

from src.adapters.vectordb.memory import MemoryIndexWriter, MemoryRetrievalEngine
from src.core.indexing import VectorSchema
from src.core.models.elements import TextChunk


def _make_chunk(object_id: str, content: str, embedding: list[float]) -> TextChunk:
    return TextChunk(
        object_id=object_id,
        doc_id="doc1",
        source_file="test.pdf",
        page_number=0,
        tool_name="mock",
        tool_version="1.0",
        content=content,
        embedding=embedding,
    )


def test_memory_backend_roundtrip_for_text():
    writer = MemoryIndexWriter(store_name="roundtrip")
    retriever = MemoryRetrievalEngine(store_name="roundtrip")
    schema = VectorSchema(dim=2, distance="cosine", multi_vector=False)

    writer.ensure_collection("text_chunks", schema)
    writer.upsert_text(
        "text_chunks",
        [
            _make_chunk("a", "hello world", [1.0, 0.0]),
            _make_chunk("b", "goodbye", [0.0, 1.0]),
        ],
    )

    results = retriever.search_text("text_chunks", [1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].element.object_id == "a"


def test_memory_backend_rejects_schema_mismatch():
    writer = MemoryIndexWriter(store_name="mismatch")
    writer.ensure_collection(
        "text_chunks",
        VectorSchema(dim=2, distance="cosine", multi_vector=False),
    )

    try:
        writer.ensure_collection(
            "text_chunks",
            VectorSchema(dim=3, distance="cosine", multi_vector=False),
        )
    except ValueError as exc:
        assert "expected" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("schema mismatch should have raised")
