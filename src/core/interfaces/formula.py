from __future__ import annotations

from typing import Protocol

from PIL.Image import Image

from ..models.document import BoundingBox
from ..models.elements import Formula


class FormulaExtractor(Protocol):
    """Recognizes and transcribes mathematical formulas to LaTeX.

    Used either standalone (on cropped images) or as an enrichment step
    after a StructuredParser has located formula regions.
    """

    @property
    def tool_name(self) -> str: ...

    @property
    def tool_version(self) -> str: ...

    def extract(
        self,
        image: Image,
        bbox: BoundingBox | None = None,
        doc_id: str = "",
        page_number: int = 0,
    ) -> list[Formula]:
        """Recognize formula(s) in an image (or image region).

        Returns one Formula per detected formula in the image.
        """
        ...
