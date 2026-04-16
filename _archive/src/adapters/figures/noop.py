from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.registry import register_figure_descriptor

if TYPE_CHECKING:
    from PIL.Image import Image


@register_figure_descriptor("noop")
class NoopFigureDescriptor:
    """Figure descriptor that skips description generation.

    Use this when no VLM is available or when figure description is not needed.
    Figures are still indexed with an empty description field.
    """

    TOOL_NAME = "noop"
    TOOL_VERSION = "1.0"

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    @property
    def tool_version(self) -> str:
        return self.TOOL_VERSION

    def describe(self, image: Image) -> str:
        return ""
