from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Pixel or point coordinates of an element on a page.

    Uses top-left origin (x0, y0) → bottom-right (x1, y1).
    Coordinates are in the coordinate space of the rendered image.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class DocumentMeta(BaseModel):
    """Metadata recorded at document ingest time."""

    doc_id: str
    source_file: str
    total_pages: int
    file_size_bytes: int
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict = Field(default_factory=dict)
