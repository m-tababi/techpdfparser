from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models.elements import Figure, Formula, Table


class StructuredParser(Protocol):
    """Extracts structured elements (tables, formulas, figures) from a PDF.

    Implementations wrap tools like MinerU, Docling, or custom detectors.
    The three element types are returned separately so downstream enrichment
    (FormulaExtractor, FigureDescriptor) can be applied selectively.
    """

    @property
    def tool_name(self) -> str: ...

    @property
    def tool_version(self) -> str: ...

    def parse(
        self, pdf_path: Path, doc_id: str
    ) -> tuple[list[Table], list[Formula], list[Figure]]:
        """Extract all structured elements from the document.

        Returns (tables, formulas, figures) as three separate lists.
        Bounding boxes and page numbers must be set on each element.
        """
        ...
