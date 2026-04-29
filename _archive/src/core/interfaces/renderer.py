from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL.Image import Image


@runtime_checkable
class PageRenderer(Protocol):
    """Renders PDF pages to PIL Images.

    Swap the adapter to change the backend without touching pipeline code.
    Example alternatives: pdf2image (poppler), pypdfium2.
    """

    @property
    def tool_name(self) -> str: ...

    @property
    def tool_version(self) -> str: ...

    def page_count(self, pdf_path: Path) -> int:
        """Return total page count without rendering any pages."""
        ...

    def render_page(self, pdf_path: Path, page_number: int) -> Image:
        """Render a single zero-indexed page to a PIL Image."""
        ...

    def render_all(self, pdf_path: Path) -> list[Image]:
        """Render all pages and return them as an ordered list."""
        ...
