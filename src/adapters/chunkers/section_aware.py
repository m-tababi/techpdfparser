from __future__ import annotations

import hashlib
from collections.abc import Iterable

from ...core.models.elements import TextChunk
from ...core.registry import register_text_chunker
from ...utils.ids import generate_element_id

CHUNK_SIZE_DEFAULT = 512
CHUNK_OVERLAP_DEFAULT = 64


@register_text_chunker("section_aware")
class SectionAwareChunker:
    """Chunker that never splits across section boundaries.

    Consecutive blocks on the same page with the same section_path are merged
    up to chunk_size. A section or page change always starts a new chunk.
    Single blocks larger than chunk_size are split with overlap.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        chunk_overlap: int = CHUNK_OVERLAP_DEFAULT,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text_blocks: list[TextChunk]) -> list[TextChunk]:
        """Split blocks into section-respecting chunks."""
        chunks: list[TextChunk] = []
        for group in self._iter_groups(text_blocks):
            chunks.extend(self._chunk_group(group))
        return chunks

    def _iter_groups(self, text_blocks: list[TextChunk]) -> Iterable[list[TextChunk]]:
        current: list[TextChunk] = []
        current_key: tuple[int, tuple[str, ...]] | None = None

        for block in text_blocks:
            key = (block.page_number, tuple(block.section_path))
            if current_key is None or key == current_key:
                current.append(block)
                current_key = key
                continue

            yield current
            current = [block]
            current_key = key

        if current:
            yield current

    def _chunk_group(self, blocks: list[TextChunk]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        current_blocks: list[TextChunk] = []
        current_length = 0

        for block in blocks:
            if len(block.content) > self._chunk_size:
                if current_blocks:
                    chunks.append(self._merge_blocks(current_blocks))
                    current_blocks = []
                    current_length = 0
                chunks.extend(self._split_block(block))
                continue

            addition = len(block.content) + (2 if current_blocks else 0)
            if current_blocks and current_length + addition > self._chunk_size:
                chunks.append(self._merge_blocks(current_blocks))
                current_blocks = [block]
                current_length = len(block.content)
                continue

            current_blocks.append(block)
            current_length += addition

        if current_blocks:
            chunks.append(self._merge_blocks(current_blocks))

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

    def _merge_blocks(self, blocks: list[TextChunk]) -> TextChunk:
        if len(blocks) == 1:
            return blocks[0]

        first = blocks[0]
        return TextChunk(
            object_id=generate_element_id(
                first.doc_id,
                first.page_number,
                "text_chunk",
                first.tool_name,
                int(
                    hashlib.sha256(
                        "|".join(block.object_id for block in blocks).encode("utf-8")
                    ).hexdigest()[:8],
                    16,
                ),
            ),
            doc_id=first.doc_id,
            source_file=first.source_file,
            page_number=first.page_number,
            tool_name=first.tool_name,
            tool_version=first.tool_version,
            content="\n\n".join(block.content for block in blocks),
            bbox=None,
            child_ids=[block.object_id for block in blocks],
            section_title=first.section_title,
            section_path=list(first.section_path),
            heading_level=first.heading_level,
        )
