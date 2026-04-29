from __future__ import annotations

from typing import Protocol

from ..models.elements import TextChunk


class TextChunker(Protocol):
    """Splits raw text blocks into retrieval-sized chunks.

    Implementations range from simple fixed-size splitters to
    semantic chunkers that respect sentence and paragraph boundaries.
    """

    def chunk(self, text_blocks: list[TextChunk]) -> list[TextChunk]:
        """Split and optionally merge blocks into retrieval-sized chunks.

        Must preserve page_number and bbox from source blocks where possible.
        Sets parent_id on child chunks that originate from a single block.
        """
        ...
