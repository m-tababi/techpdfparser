from __future__ import annotations

from pathlib import Path

from ...core.models.elements import TextChunk
from ...core.registry import register_text_extractor
from ...utils.ids import generate_element_id


@register_text_extractor("pymupdf_text")
class PyMuPDFTextExtractor:
    """Text extractor using PyMuPDF's native text layer.

    No model required — reads the embedded text directly from the PDF.
    Suitable for digitally-born PDFs; scanned documents need an OCR extractor.
    Much faster than olmOCR2 and requires no GPU.

    Requires: pip install pymupdf (already a dependency)
    """

    TOOL_NAME = "pymupdf_text"
    TOOL_VERSION = "1.24"

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    @property
    def tool_version(self) -> str:
        return self.TOOL_VERSION

    def extract_page(
        self, pdf_path: Path, page_number: int, doc_id: str
    ) -> list[TextChunk]:
        """Extract text blocks from one page. One TextChunk per block."""
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            page = doc[page_number]
            blocks = page.get_text("blocks")
            return self._blocks_to_chunks(blocks, doc_id, page_number, str(pdf_path))
        finally:
            doc.close()

    def extract_all(self, pdf_path: Path, doc_id: str) -> list[TextChunk]:
        """Extract all text blocks from every page in one pass."""
        import fitz

        doc = fitz.open(str(pdf_path))
        chunks: list[TextChunk] = []
        try:
            for page_number, page in enumerate(doc):
                blocks = page.get_text("blocks")
                chunks.extend(
                    self._blocks_to_chunks(blocks, doc_id, page_number, str(pdf_path))
                )
        finally:
            doc.close()
        return chunks

    def _blocks_to_chunks(
        self,
        blocks: list,
        doc_id: str,
        page_number: int,
        source_file: str,
    ) -> list[TextChunk]:
        """Convert fitz block tuples to TextChunk objects.

        fitz block tuple: (x0, y0, x1, y1, text, block_no, block_type)
        block_type 0 = text, 1 = image — skip non-text and blank blocks.
        """
        chunks: list[TextChunk] = []
        for seq, block in enumerate(blocks):
            block_type = block[6]
            text = block[4].strip()
            if block_type != 0 or not text:
                continue
            chunks.append(
                TextChunk(
                    object_id=generate_element_id(
                        doc_id, page_number, "text_chunk", self.TOOL_NAME, seq
                    ),
                    doc_id=doc_id,
                    source_file=source_file,
                    page_number=page_number,
                    tool_name=self.TOOL_NAME,
                    tool_version=self.TOOL_VERSION,
                    content=text,
                )
            )
        return chunks
