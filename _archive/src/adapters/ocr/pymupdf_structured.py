from __future__ import annotations

from pathlib import Path

from ...core.models.document import BoundingBox
from ...core.models.elements import TextChunk
from ...core.registry import register_text_extractor
from ...utils.ids import generate_element_id
from ...utils.sections import (
    SectionMarker,
    assign_sections,
    detect_sections_from_fonts,
    detect_sections_from_toc,
)


@register_text_extractor("pymupdf_structured")
class PyMuPDFStructuredExtractor:
    """Text extractor that preserves document structure via font-size analysis.

    Uses page.get_text("dict") to access span-level font metadata, detects
    headings via PyMuPDF TOC (ground truth when available) or font-size
    heuristics (fallback), then annotates every TextChunk with section context.

    Designed as a drop-in replacement for pymupdf_text — same protocol,
    richer output.
    """

    TOOL_NAME = "pymupdf_structured"
    TOOL_VERSION = "1.0"

    def __init__(
        self,
        heading_size_ratio: float = 1.3,
        max_heading_levels: int = 4,
    ) -> None:
        self._heading_size_ratio = heading_size_ratio
        self._max_heading_levels = max_heading_levels

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    @property
    def tool_version(self) -> str:
        return self.TOOL_VERSION

    def extract_page(self, pdf_path: Path, page_number: int, doc_id: str) -> list[TextChunk]:
        """Extract structured text blocks from one page."""
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            markers = self._build_markers(doc)
            page = doc[page_number]
            chunks = self._page_to_chunks(page, page_number, doc_id, str(pdf_path))
            assign_sections(chunks, markers)
            return chunks
        finally:
            doc.close()

    def extract_all(self, pdf_path: Path, doc_id: str) -> list[TextChunk]:
        """Extract structured text blocks from every page in one pass."""
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            markers = self._build_markers(doc)
            chunks: list[TextChunk] = []
            for page_number, page in enumerate(doc):
                chunks.extend(self._page_to_chunks(page, page_number, doc_id, str(pdf_path)))
            assign_sections(chunks, markers)
            return chunks
        finally:
            doc.close()

    def get_markers(self, pdf_path: Path) -> list[SectionMarker]:
        """Return section markers for a PDF without extracting text.

        Used by the text pipeline to persist sections.json independently.
        """
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            return self._build_markers(doc)
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_markers(self, doc: object) -> list[SectionMarker]:
        """TOC-first detection; fall back to font heuristics."""
        markers = detect_sections_from_toc(doc)
        if markers:
            return markers
        all_spans = self._collect_spans(doc)
        return detect_sections_from_fonts(
            all_spans,
            size_ratio=self._heading_size_ratio,
            max_levels=self._max_heading_levels,
        )

    def _collect_spans(self, doc: object) -> list[dict]:
        """Gather all spans across all pages, injecting page number."""
        spans: list[dict] = []
        for page_number, page in enumerate(doc):  # type: ignore[call-overload]
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 = text
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span["page"] = page_number
                        spans.append(span)
        return spans

    def _page_to_chunks(
        self,
        page: object,
        page_number: int,
        doc_id: str,
        source_file: str,
    ) -> list[TextChunk]:
        """Convert fitz dict-mode blocks into TextChunk objects with bbox."""
        page_dict = page.get_text("dict")  # type: ignore[attr-defined]
        chunks: list[TextChunk] = []
        seq = 0
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            text = self._block_text(block).strip()
            if not text:
                continue
            bbox = self._block_bbox(block)
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
                    bbox=bbox,
                )
            )
            seq += 1
        return chunks

    def _block_text(self, block: dict) -> str:
        lines = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if line_text.strip():
                lines.append(line_text)
        return "\n".join(lines)

    def _block_bbox(self, block: dict) -> BoundingBox | None:
        bbox = block.get("bbox")
        if bbox is None:
            return None
        return BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])
