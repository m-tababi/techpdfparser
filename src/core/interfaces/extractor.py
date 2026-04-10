from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models.elements import TextChunk


class TextExtractor(Protocol):
    """Extracts text from PDF pages into TextChunk objects.

    Each implementation wraps a different extraction strategy:
    native PDF text layer, OCR, markdown-aware extraction, etc.
    Swap adapters to compare tool quality on the same document set.
    """

    @property
    def tool_name(self) -> str: ...

    @property
    def tool_version(self) -> str: ...

    def extract_page(self, pdf_path: Path, page_number: int, doc_id: str) -> list[TextChunk]:
        """Extract raw text blocks from a single page.

        Returns un-chunked blocks. The TextChunker handles splitting downstream.
        """
        ...

    def extract_all(self, pdf_path: Path, doc_id: str) -> list[TextChunk]:
        """Extract text from all pages in reading order."""
        ...
