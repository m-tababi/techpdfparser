from __future__ import annotations

from pathlib import Path

from ...core.models.document import BoundingBox
from ...core.models.elements import Figure, Formula, Table
from ...core.registry import register_structured_parser
from ...utils.ids import generate_element_id


@register_structured_parser("mineru25")
class MinerU25Parser:
    """Structured parser using MinerU 2.5-Pro.

    MinerU detects and extracts tables, formulas, and figures with bounding
    boxes. Its output is then enriched by FormulaExtractor and FigureDescriptor
    in the StructuredPipeline.

    Replace with Docling or another parser by registering under a different name.

    Model: opendatalab/MinerU
    Requires: pip install mineru
    """

    TOOL_NAME = "mineru25"
    TOOL_VERSION = "2.5"

    def __init__(self, model_path: str = "", device: str = "cuda") -> None:
        self._model_path = model_path
        self._device = device
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from mineru.pipeline import PDFPipeline

            self._pipeline = PDFPipeline(
                model_path=self._model_path, device=self._device
            )
        except ImportError:
            raise ImportError("mineru not installed. Run: pip install mineru")

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    @property
    def tool_version(self) -> str:
        return self.TOOL_VERSION

    def parse(
        self, pdf_path: Path, doc_id: str
    ) -> tuple[list[Table], list[Formula], list[Figure]]:
        self._load()
        raw = self._pipeline.run(str(pdf_path))
        source = str(pdf_path)
        tables = self._to_tables(raw, doc_id, source)
        formulas = self._to_formulas(raw, doc_id, source)
        figures = self._to_figures(raw, doc_id, source)
        return tables, formulas, figures

    def _to_tables(self, raw: dict, doc_id: str, source: str) -> list[Table]:
        tables: list[Table] = []
        for i, item in enumerate(raw.get("tables", [])):
            tables.append(
                Table(
                    object_id=generate_element_id(
                        doc_id, item.get("page", 0), "table", self.TOOL_NAME, i
                    ),
                    doc_id=doc_id,
                    source_file=source,
                    page_number=item.get("page", 0),
                    tool_name=self.TOOL_NAME,
                    tool_version=self.TOOL_VERSION,
                    bbox=_to_bbox(item.get("bbox")),
                    content=item.get("markdown", ""),
                    rows=item.get("rows", []),
                    headers=item.get("headers", []),
                )
            )
        return tables

    def _to_formulas(self, raw: dict, doc_id: str, source: str) -> list[Formula]:
        formulas: list[Formula] = []
        for i, item in enumerate(raw.get("formulas", [])):
            formulas.append(
                Formula(
                    object_id=generate_element_id(
                        doc_id, item.get("page", 0), "formula", self.TOOL_NAME, i
                    ),
                    doc_id=doc_id,
                    source_file=source,
                    page_number=item.get("page", 0),
                    tool_name=self.TOOL_NAME,
                    tool_version=self.TOOL_VERSION,
                    bbox=_to_bbox(item.get("bbox")),
                    latex=item.get("latex", ""),
                    content=item.get("text", item.get("latex", "")),
                )
            )
        return formulas

    def _to_figures(self, raw: dict, doc_id: str, source: str) -> list[Figure]:
        figures: list[Figure] = []
        for i, item in enumerate(raw.get("figures", [])):
            figures.append(
                Figure(
                    object_id=generate_element_id(
                        doc_id, item.get("page", 0), "figure", self.TOOL_NAME, i
                    ),
                    doc_id=doc_id,
                    source_file=source,
                    page_number=item.get("page", 0),
                    tool_name=self.TOOL_NAME,
                    tool_version=self.TOOL_VERSION,
                    bbox=_to_bbox(item.get("bbox")),
                    image_path=item.get("image_path", ""),
                    caption=item.get("caption"),
                )
            )
        return figures


def _to_bbox(raw: list | None) -> BoundingBox | None:
    if not raw or len(raw) < 4:
        return None
    return BoundingBox(x0=raw[0], y0=raw[1], x1=raw[2], y1=raw[3])
