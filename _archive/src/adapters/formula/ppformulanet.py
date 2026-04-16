from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.models.document import BoundingBox
from ...core.models.elements import Formula
from ...core.registry import register_formula_extractor
from ...utils.ids import generate_element_id

if TYPE_CHECKING:
    from PIL.Image import Image


@register_formula_extractor("ppformulanet")
class PPFormulaNetExtractor:
    """Formula recognizer using PP-FormulaNet (PaddleOCR formula model).

    Converts formula image crops to LaTeX. Use as a standalone enrichment
    step or when the StructuredParser does not provide LaTeX output.

    Replace with Pix2Tex, UniMER-Net, or any other formula recognizer by
    registering under a different name and updating the config.

    Requires: pip install paddlepaddle paddleocr
    """

    TOOL_NAME = "ppformulanet"
    TOOL_VERSION = "1.0"

    def __init__(self, model_path: str = "", device: str = "cuda") -> None:
        self._model_path = model_path
        self._device = device
        self._ocr = None

    def _load(self) -> None:
        if self._ocr is not None:
            return
        try:
            from paddleocr import PPStructure

            self._ocr = PPStructure(formula=True, show_log=False)
        except ImportError:
            raise ImportError(
                "paddleocr not installed. Run: pip install paddlepaddle paddleocr"
            )

    def unload(self) -> None:
        self._ocr = None

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
        self._load()
        import numpy as np

        img_array = np.array(image)
        results = self._ocr(img_array)
        return [
            self._to_formula(r, doc_id, page_number, i)
            for i, r in enumerate(results)
            if r.get("type") == "formula"
        ]

    def _to_formula(
        self, raw: dict, doc_id: str, page_number: int, seq: int
    ) -> Formula:
        latex = raw.get("res", {}).get("latex", "")
        return Formula(
            object_id=generate_element_id(
                doc_id, page_number, "formula", self.TOOL_NAME, seq
            ),
            doc_id=doc_id,
            source_file="",
            page_number=page_number,
            tool_name=self.TOOL_NAME,
            tool_version=self.TOOL_VERSION,
            latex=latex,
            content=latex,
        )
