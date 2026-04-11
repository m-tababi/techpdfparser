from __future__ import annotations

from ...core.models.elements import TextChunk
from ...core.registry import register_text_chunker
from ...utils.ids import generate_element_id

CHUNK_SIZE_DEFAULT = 512
CHUNK_OVERLAP_DEFAULT = 64


@register_text_chunker("section_aware")
class SectionAwareChunker:
    """Chunker that never splits across section boundaries.

    Blocks that share a section_title are chunked together up to chunk_size.
    A new section always starts a new chunk even if the previous one is short.
    Within a section, large blocks are split with overlap (same as FixedSizeChunker).
    Section metadata (section_title, section_path, heading_level) is inherited
    by every chunk produced from a block.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        chunk_overlap: int = CHUNK_OVERLAP_DEFAULT,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text_blocks: list[TextChunk]) -> list[TextChunk]:
        """Split blocks into section-respecting chunks."""
        chunks: list[TextChunk] = []
        for block in text_blocks:
            chunks.extend(self._split_block(block))
        return chunks

    def _split_block(self, block: TextChunk) -> list[TextChunk]:
        """Split one block. Blocks <= chunk_size are returned as-is."""
        text = block.content
        if len(text) <= self._chunk_size:
            return [block]

        chunks: list[TextChunk] = []
        start = 0
        seq = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(self._make_chunk(block, chunk_text, start, end, seq))
            start += self._chunk_size - self._chunk_overlap
            seq += 1

        return chunks

    def _make_chunk(
        self,
        source: TextChunk,
        text: str,
        char_start: int,
        char_end: int,
        seq: int,
    ) -> TextChunk:
        chunk = TextChunk(
            object_id=generate_element_id(
                source.doc_id,
                source.page_number,
                "text_chunk",
                source.tool_name,
                seq,
            ),
            doc_id=source.doc_id,
            source_file=source.source_file,
            page_number=source.page_number,
            tool_name=source.tool_name,
            tool_version=source.tool_version,
            content=text,
            char_start=char_start,
            char_end=char_end,
            parent_id=source.object_id,
            bbox=source.bbox,
        )
        # Propagate section context from the source block
        chunk.section_title = source.section_title
        chunk.section_path = list(source.section_path)
        chunk.heading_level = source.heading_level
        return chunk
