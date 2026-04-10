from __future__ import annotations

from pathlib import Path

import PIL.Image

from ...core.registry import register_renderer


@register_renderer("pymupdf")
class PyMuPDFRenderer:
    """Renders PDF pages to PIL Images using PyMuPDF (fitz).

    PyMuPDF is fast, handles most PDF features, and requires no external
    binary dependencies. Increase `dpi` for higher fidelity at the cost
    of larger images and more memory during embedding.

    Replace with pdf2image (poppler) or pypdfium2 for different rendering quality.
    Requires: pip install pymupdf
    """

    TOOL_NAME = "pymupdf"
    TOOL_VERSION = "1.24"

    def __init__(self, dpi: int = 150) -> None:
        # 150 DPI balances readability for VLMs vs. memory footprint.
        # 72 DPI is the internal PDF unit; the matrix scales to target DPI.
        self._dpi = dpi
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

    @property
    def tool_version(self) -> str:
        return self.TOOL_VERSION

    def page_count(self, pdf_path: Path) -> int:
        with self._fitz.open(str(pdf_path)) as doc:
            return doc.page_count

    def render_page(self, pdf_path: Path, page_number: int) -> PIL.Image.Image:
        with self._fitz.open(str(pdf_path)) as doc:
            page = doc[page_number]
            mat = self._fitz.Matrix(self._dpi / 72, self._dpi / 72)
            pixmap = page.get_pixmap(matrix=mat, alpha=False)
            return PIL.Image.frombytes(
                "RGB", [pixmap.width, pixmap.height], pixmap.samples
            )

    def render_all(self, pdf_path: Path) -> list[PIL.Image.Image]:
        count = self.page_count(pdf_path)
        return [self.render_page(pdf_path, i) for i in range(count)]
