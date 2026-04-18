"""CPU-only baseline segmenter using PyMuPDF's native text block layout.

Proves the Segmenter Protocol is swappable without a GPU.
Emits only ElementType.TEXT regions — no heading/table/formula detection.
For richer segmentation, use the MinerU adapter.
"""
from __future__ import annotations

from pathlib import Path

from ..models import ElementContent, ElementType, Region
from ..registry import register_segmenter


@register_segmenter("pymupdf_text")
class PyMuPDFTextSegmenter:
    TOOL_NAME = "pymupdf_text"

    def __init__(self) -> None:
        self._fitz = self._import_fitz()

    @staticmethod
    def _import_fitz():
        try:
            import fitz
            return fitz
        except ImportError:
            raise ImportError("pymupdf not installed. Run: pip install pymupdf")

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def segment(self, pdf_path: Path) -> list[Region]:
        regions: list[Region] = []
        with self._fitz.open(str(pdf_path)) as doc:
            for page_idx, page in enumerate(doc):
                data = page.get_text("dict")
                for block in data.get("blocks", []):
                    # Block type 0 = text block; type 1 = image (ignored here).
                    if block.get("type", 0) != 0:
                        continue
                    text = self._block_text(block)
                    if not text.strip():
                        continue
                    bbox = list(block.get("bbox", [0.0, 0.0, 0.0, 0.0]))
                    regions.append(
                        Region(
                            page=page_idx,
                            bbox=bbox,
                            region_type=ElementType.TEXT,
                            confidence=1.0,
                            content=ElementContent(text=text),
                        )
                    )
        return regions

    @staticmethod
    def _block_text(block: dict) -> str:
        lines: list[str] = []
        for line in block.get("lines", []):
            spans = [s.get("text", "") for s in line.get("spans", [])]
            joined = "".join(spans).strip()
            if joined:
                lines.append(joined)
        return "\n".join(lines)
