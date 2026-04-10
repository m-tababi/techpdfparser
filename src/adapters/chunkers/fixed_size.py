from __future__ import annotations

from ...core.models.elements import TextChunk
from ...core.registry import register_text_chunker
from ...utils.ids import generate_element_id


@register_text_chunker("fixed_size")
class FixedSizeChunker:
    """Splits text blocks into fixed-size overlapping chunks by character count.

    This is a baseline implementation. Replace with a semantic chunker (sentence
    splitter, paragraph detector) by registering under a different name and
    updating `pipelines.text.chunker` in config.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text_blocks: list[TextChunk]) -> list[TextChunk]:
        """Split blocks into fixed-size chunks, preserving page metadata."""
        chunks: list[TextChunk] = []
        for block in text_blocks:
            chunks.extend(self._split_block(block))
        return chunks

    def _split_block(self, block: TextChunk) -> list[TextChunk]:
        text = block.content
        if len(text) <= self._chunk_size:
            return [block]

        chunks: list[TextChunk] = []
        start = 0
        seq = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(
                TextChunk(
                    object_id=generate_element_id(
                        block.doc_id,
                        block.page_number,
                        "text_chunk",
                        block.tool_name,
                        seq,
                    ),
                    doc_id=block.doc_id,
                    source_file=block.source_file,
                    page_number=block.page_number,
                    tool_name=block.tool_name,
                    tool_version=block.tool_version,
                    content=chunk_text,
                    char_start=start,
                    char_end=end,
                    parent_id=block.object_id,
                )
            )
            start += self._chunk_size - self._chunk_overlap
            seq += 1

        return chunks
