"""Tests for the section-aware chunker."""

import pytest

from src.adapters.chunkers.section_aware import SectionAwareChunker
from src.core.models.elements import TextChunk


def make_block(
    content: str,
    object_id: str = "block_0",
    page: int = 0,
    section_title: str | None = None,
    section_path: list[str] | None = None,
    heading_level: int | None = None,
) -> TextChunk:
    block = TextChunk(
        object_id=object_id,
        doc_id="doc1",
        source_file="test.pdf",
        page_number=page,
        tool_name="test",
        tool_version="1.0",
        content=content,
    )
    block.section_title = section_title
    block.section_path = section_path or []
    block.heading_level = heading_level
    return block


class TestSectionAwareChunker:
    def test_rejects_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            SectionAwareChunker(chunk_size=0, chunk_overlap=0)

    def test_rejects_invalid_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            SectionAwareChunker(chunk_size=10, chunk_overlap=10)

    def test_short_block_unchanged(self):
        chunker = SectionAwareChunker(chunk_size=100, chunk_overlap=0)
        block = make_block("Short text", section_title="Intro")
        chunks = chunker.chunk([block])
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"

    def test_splits_long_block(self):
        chunker = SectionAwareChunker(chunk_size=10, chunk_overlap=0)
        block = make_block("A" * 30)
        chunks = chunker.chunk([block])
        assert len(chunks) == 3

    def test_inherits_section_title(self):
        chunker = SectionAwareChunker(chunk_size=10, chunk_overlap=0)
        block = make_block("A" * 20, section_title="Methods", section_path=["Methods"])
        chunks = chunker.chunk([block])
        for chunk in chunks:
            assert chunk.section_title == "Methods"
            assert chunk.section_path == ["Methods"]

    def test_inherits_heading_level(self):
        chunker = SectionAwareChunker(chunk_size=10, chunk_overlap=0)
        block = make_block("A" * 20, heading_level=2)
        chunks = chunker.chunk([block])
        assert all(c.heading_level == 2 for c in chunks)

    def test_char_positions_set(self):
        chunker = SectionAwareChunker(chunk_size=5, chunk_overlap=0)
        block = make_block("ABCDEABCDE")  # exactly 2 chunks
        chunks = chunker.chunk([block])
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 5
        assert chunks[1].char_start == 5
        assert chunks[1].char_end == 10

    def test_parent_id_set_on_splits(self):
        chunker = SectionAwareChunker(chunk_size=5, chunk_overlap=0)
        block = make_block("ABCDE" * 2)
        chunks = chunker.chunk([block])
        for chunk in chunks:
            assert chunk.parent_id == block.object_id

    def test_none_section_propagated(self):
        chunker = SectionAwareChunker(chunk_size=5, chunk_overlap=0)
        block = make_block("ABCDE" * 2, section_title=None)
        chunks = chunker.chunk([block])
        assert all(c.section_title is None for c in chunks)

    def test_multiple_blocks_with_same_section_are_merged(self):
        chunker = SectionAwareChunker(chunk_size=100, chunk_overlap=0)
        blocks = [
            make_block(
                "text in intro",
                object_id="block_1",
                section_title="Intro",
                section_path=["Intro"],
            ),
            make_block(
                "more intro text",
                object_id="block_2",
                section_title="Intro",
                section_path=["Intro"],
            ),
        ]
        chunks = chunker.chunk(blocks)
        assert len(chunks) == 1
        assert chunks[0].section_title == "Intro"
        assert chunks[0].child_ids == ["block_1", "block_2"]
        assert chunks[0].bbox is None

    def test_section_change_starts_new_chunk(self):
        chunker = SectionAwareChunker(chunk_size=100, chunk_overlap=0)
        blocks = [
            make_block(
                "text in intro",
                object_id="block_1",
                section_title="Intro",
                section_path=["Intro"],
            ),
            make_block(
                "text in methods",
                object_id="block_2",
                section_title="Methods",
                section_path=["Methods"],
            ),
        ]
        chunks = chunker.chunk(blocks)
        assert len(chunks) == 2
        assert chunks[0].section_title == "Intro"
        assert chunks[1].section_title == "Methods"

    def test_page_change_starts_new_chunk_even_with_same_section(self):
        chunker = SectionAwareChunker(chunk_size=100, chunk_overlap=0)
        blocks = [
            make_block(
                "page zero",
                object_id="block_1",
                page=0,
                section_title="Intro",
                section_path=["Intro"],
            ),
            make_block(
                "page one",
                object_id="block_2",
                page=1,
                section_title="Intro",
                section_path=["Intro"],
            ),
        ]
        chunks = chunker.chunk(blocks)
        assert len(chunks) == 2
        assert chunks[0].page_number == 0
        assert chunks[1].page_number == 1

    def test_ids_unique_across_splits(self):
        chunker = SectionAwareChunker(chunk_size=5, chunk_overlap=0)
        block = make_block("ABCDE" * 4)
        chunks = chunker.chunk([block])
        ids = [c.object_id for c in chunks]
        assert len(ids) == len(set(ids))
