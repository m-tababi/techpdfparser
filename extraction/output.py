"""Write extraction results to the spec output format.

Produces content_list.json, document_rich.json, and image crops
inside an output directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL.Image import Image

from .models import ContentList, DocumentRich


class OutputWriter:
    """Manages writing extraction output to a directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_content_list(self, content_list: ContentList) -> Path:
        path = self.output_dir / "content_list.json"
        data = content_list.model_dump(mode="json", exclude_none=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def write_document_rich(self, document_rich: DocumentRich) -> Path:
        path = self.output_dir / "document_rich.json"
        data = document_rich.model_dump(mode="json", exclude_none=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def write_segmentation(self, regions: list) -> Path:
        """Write raw segmentation regions for later inspection."""
        path = self.output_dir / "segmentation.json"
        data = [r.model_dump(mode="json", exclude_none=True) for r in regions]
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def save_page_image(self, page: int, image: Image) -> Path:
        page_dir = self.output_dir / "pages" / str(page)
        page_dir.mkdir(parents=True, exist_ok=True)
        path = page_dir / "page.png"
        image.save(path)
        return path

    def save_element_crop(
        self, page: int, element_id: str, element_type: str, image: Image
    ) -> Path:
        page_dir = self.output_dir / "pages" / str(page)
        page_dir.mkdir(parents=True, exist_ok=True)
        path = page_dir / f"{element_id}_{element_type}.png"
        image.save(path)
        return path

    def crop_region(self, page_image: Image, bbox: list[float]) -> Image:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        return page_image.crop((x0, y0, x1, y1))
