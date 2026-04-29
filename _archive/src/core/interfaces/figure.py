from __future__ import annotations

from typing import Protocol

from PIL.Image import Image


class FigureDescriptor(Protocol):
    """Generates natural-language descriptions of figures and diagrams.

    The description is embedded as text and stored alongside the figure crop,
    enabling text-based retrieval of visual content.
    """

    @property
    def tool_name(self) -> str: ...

    @property
    def tool_version(self) -> str: ...

    def describe(self, image: Image) -> str:
        """Generate a concise technical description of a figure image.

        Should describe: visualization type, data shown, key trends or values.
        """
        ...
