"""Test fixed-size chunking logic without any ML models."""

import pytest

from src.adapters.chunkers.fixed_size import FixedSizeChunker
from src.core.models.elements import TextChunk


def make_block(content: str, page: int = 0) -> TextChunk:
    return TextChunk(
        object_id="block_0",
        doc_id="doc1",
        source_file="test.pdf",
        page_number=page,
        tool_name="test",
        tool_version="1.0",
        content=content,
    )


class TestFixedSizeChunker:
    def test_short_block_returned_unchanged(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        block = make_block("Short text")
        chunks = chunker.chunk([block])
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"

    def test_splits_long_block(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        block = make_block("0123456789" * 3)  # 30 chars
        chunks = chunker.chunk([block])
        assert len(chunks) == 3
        assert all(len(c.content) <= 10 for c in chunks)

    def test_overlap_is_applied(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=5)
        block = make_block("A" * 20)
        chunks = chunker.chunk([block])
        # With overlap=5, each step advances by 5; 20 chars → multiple chunks
        assert len(chunks) > 2

    def test_char_start_end_set(self):
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        block = make_block("ABCDE" * 2)  # 10 chars → 2 chunks
        chunks = chunker.chunk([block])
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 5
        assert chunks[1].char_start == 5
        assert chunks[1].char_end == 10

    def test_parent_id_set(self):
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        block = make_block("ABCDE" * 2)
        chunks = chunker.chunk([block])
        for chunk in chunks:
            assert chunk.parent_id == block.object_id

    def test_page_number_preserved(self):
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        block = make_block("ABCDE" * 2, page=3)
        chunks = chunker.chunk([block])
        assert all(c.page_number == 3 for c in chunks)

    def test_multiple_blocks(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        blocks = [make_block(f"Block {i}" * 5, page=i) for i in range(3)]
        chunks = chunker.chunk(blocks)
        assert len(chunks) == 3

    def test_ids_are_unique(self):
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        block = make_block("ABCDE" * 4)
        chunks = chunker.chunk([block])
        ids = [c.object_id for c in chunks]
        assert len(ids) == len(set(ids))
