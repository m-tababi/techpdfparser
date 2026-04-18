"""No-op extractor stubs.

Zweck: Der Router ruft für jede Region, die der Segmenter ohne Content
liefert, einen typisierten Extractor auf. Wenn der Segmenter (z. B. MinerU)
aber bereits Content mitliefert, ist der Extractor-Call ein Fallback, der
nie feuert — trotzdem muss die Pipeline einen konfigurierten Adapter bekommen.
Die Stubs halten diesen Pfad leichtgewichtig und GPU-frei.
"""
from __future__ import annotations

from PIL.Image import Image

from ..models import ElementContent
from ..registry import (
    register_figure_descriptor,
    register_formula_extractor,
    register_table_extractor,
    register_text_extractor,
)


@register_text_extractor("noop")
class NoopTextExtractor:
    TOOL_NAME = "noop"

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def extract(self, page_image: Image, page_number: int) -> ElementContent:
        return ElementContent(text="")


@register_table_extractor("noop")
class NoopTableExtractor:
    TOOL_NAME = "noop"

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        return ElementContent()


@register_formula_extractor("noop")
class NoopFormulaExtractor:
    TOOL_NAME = "noop"

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        return ElementContent()


@register_figure_descriptor("noop")
class NoopFigureDescriptor:
    TOOL_NAME = "noop"

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def describe(self, image: Image) -> str:
        return ""
