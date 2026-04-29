from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...core.models.document import BoundingBox
from ...core.models.elements import Formula
from ...core.registry import register_formula_extractor
from ...utils.ids import generate_element_id

if TYPE_CHECKING:
    from PIL.Image import Image


@register_formula_extractor("pix2tex")
class Pix2TexExtractor:
    """Formula extractor using pix2tex (LaTeX-OCR).

    CPU-compatible alternative to PP-FormulaNet. Takes a page image or a
    cropped formula region and returns its LaTeX representation.

    pix2tex prints CUDA warnings on CPU machines; we suppress them in _load()
    to avoid noisy logs when running the AMD/CPU stack.

    Requires: pip install pix2tex
    """

    TOOL_NAME = "pix2tex"
    TOOL_VERSION = "0.1"

    def __init__(self, device: str = "cpu") -> None:
        # device stored for forward-compatibility; pix2tex manages its own
        # device selection internally based on torch.cuda.is_available()
        self._device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # Suppress pix2tex's CUDA-not-found warnings — expected on AMD/CPU
        logging.getLogger("pix2tex").setLevel(logging.ERROR)
        try:
            from pix2tex.cli import LatexOCR

            self._model = LatexOCR()
        except ImportError:
            raise ImportError(
                "pix2tex not installed. Run: pip install pix2tex"
            )

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    @property
    def tool_version(self) -> str:
        return self.TOOL_VERSION

    def extract(
        self,
        image: Image,
        bbox: BoundingBox | None = None,
        doc_id: str = "",
        page_number: int = 0,
    ) -> list[Formula]:
        """Extract LaTeX from an image or a bounding-box crop of it.

        Returns [] when pix2tex fails (e.g. image contains no formula).
        Failures are expected when the caller passes figure or table images.
        """
        self._load()
        region = self._crop(image, bbox) if bbox else image
        try:
            latex = self._model(region)
        except Exception:
            return []

        if not latex or not latex.strip():
            return []

        return [self._to_formula(latex.strip(), doc_id, page_number)]

    def _crop(self, image: Image, bbox: BoundingBox) -> Image:
        return image.crop((bbox.x0, bbox.y0, bbox.x1, bbox.y1))

    def _to_formula(self, latex: str, doc_id: str, page_number: int) -> Formula:
        return Formula(
            object_id=generate_element_id(
                doc_id, page_number, "formula", self.TOOL_NAME
            ),
            doc_id=doc_id,
            source_file="",
            page_number=page_number,
            tool_name=self.TOOL_NAME,
            tool_version=self.TOOL_VERSION,
            latex=latex,
            content=latex,
        )
