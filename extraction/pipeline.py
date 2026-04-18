"""Extraction pipeline orchestration.

Flow: render pages -> segment -> route regions -> extract -> merge -> write output.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL.Image import Image

from .interfaces import (
    FigureDescriptor,
    FormulaExtractor,
    PageRenderer,
    Segmenter,
    TableExtractor,
    TextExtractor,
)
from .models import (
    ContentList,
    Element,
    ElementContent,
    ElementType,
    Region,
)
from .output import OutputWriter

# Types that get visual crops saved
_VISUAL_TYPES = {
    ElementType.TABLE,
    ElementType.FORMULA,
    ElementType.FIGURE,
    ElementType.DIAGRAM,
    ElementType.TECHNICAL_DRAWING,
}

# Map region types to which extractor handles them
_TEXT_TYPES = {ElementType.TEXT, ElementType.HEADING}


class ExtractionPipeline:
    """Orchestrates the full extraction flow for one PDF."""

    def __init__(
        self,
        renderer: PageRenderer,
        segmenter: Segmenter,
        text_extractor: TextExtractor,
        table_extractor: TableExtractor,
        formula_extractor: FormulaExtractor,
        figure_descriptor: FigureDescriptor,
        output_dir: Path,
        confidence_threshold: float = 0.3,
        dpi: int = 150,
    ) -> None:
        self.renderer = renderer
        self.segmenter = segmenter
        self.text_extractor = text_extractor
        self.table_extractor = table_extractor
        self.formula_extractor = formula_extractor
        self.figure_descriptor = figure_descriptor
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.dpi = dpi

    def run(self, pdf_path: Path) -> ContentList:
        """Run the full extraction pipeline on a single PDF."""
        self._assert_output_dir_clean()
        writer = OutputWriter(self.output_dir)

        # 1. Render all pages
        page_count = self.renderer.page_count(pdf_path)
        page_images: list[Image] = []
        for i in range(page_count):
            img = self.renderer.render_page(pdf_path, i)
            writer.save_page_image(page=i, image=img)
            page_images.append(img)

        # 2. Segment
        regions = self.segmenter.segment(pdf_path)
        writer.write_segmentation(regions)

        doc_id = self._make_doc_id(pdf_path)

        # 3. Route and extract
        elements: list[Element] = []
        for idx, region in enumerate(regions):
            content = self._extract_region(region, page_images)
            if content is None:
                continue
            if self._is_droppable(region.region_type, content):
                continue

            element_id = self._make_element_id(doc_id, region)
            extractor_name = self._extractor_for(region)

            el = Element(
                element_id=element_id,
                type=region.region_type,
                page=region.page,
                bbox=region.bbox,
                reading_order_index=idx,
                section_path=[],
                confidence=region.confidence,
                extractor=extractor_name,
                content=content,
            )
            elements.append(el)

        # 4. Confidence filter
        elements = [e for e in elements if e.confidence >= self.confidence_threshold]

        # 5. Save visual crops
        for el in elements:
            if el.type in _VISUAL_TYPES and 0 <= el.page < len(page_images):
                crop = writer.crop_region(page_images[el.page], el.bbox, dpi=self.dpi)
                rel_path = writer.save_element_crop(
                    page=el.page,
                    element_id=el.element_id,
                    element_type=el.type.value,
                    image=crop,
                )
                el.content.image_path = str(
                    rel_path.relative_to(self.output_dir)
                )

        # 5b. Drop visuals that ended up with neither image_path nor description.
        elements = [
            e for e in elements
            if e.type not in _VISUAL_TYPES
            or e.content.image_path
            or (e.content.description or "").strip()
        ]

        # 6. Reassign reading order after filtering
        for idx, el in enumerate(elements):
            el.reading_order_index = idx

        # 7. Write per-element sidecars (source of truth)
        for el in elements:
            writer.write_element_sidecar(el)

        # 8. Build content_list.json deterministically from the sidecars
        content_list = writer.build_content_list(
            doc_id=doc_id,
            source_file=pdf_path.name,
            total_pages=page_count,
            segmentation_tool=self.segmenter.tool_name,
        )
        writer.write_content_list(content_list)

        return content_list

    def _is_droppable(self, region_type: ElementType, content: ElementContent) -> bool:
        if region_type in _TEXT_TYPES:
            return not (content.text or "").strip()
        if region_type == ElementType.TABLE:
            return not (content.markdown or content.text)
        if region_type == ElementType.FORMULA:
            return not (content.latex or content.text)
        # Visual types: drop happens after image cropping (step 5 in run()), not here.
        return False

    def _extract_region(
        self, region: Region, page_images: list[Image]
    ) -> ElementContent | None:
        # If segmenter already extracted content, use it
        if region.content is not None:
            return region.content

        if region.page < 0 or region.page >= len(page_images):
            return None

        page_img = page_images[region.page]

        if region.region_type in _TEXT_TYPES:
            return self.text_extractor.extract(page_img, region.page)
        elif region.region_type == ElementType.TABLE:
            crop = OutputWriter(self.output_dir).crop_region(
                page_img, region.bbox, dpi=self.dpi
            )
            return self.table_extractor.extract(crop, region.page)
        elif region.region_type == ElementType.FORMULA:
            crop = OutputWriter(self.output_dir).crop_region(
                page_img, region.bbox, dpi=self.dpi
            )
            return self.formula_extractor.extract(crop, region.page)
        elif region.region_type in {
            ElementType.FIGURE,
            ElementType.DIAGRAM,
            ElementType.TECHNICAL_DRAWING,
        }:
            crop = OutputWriter(self.output_dir).crop_region(
                page_img, region.bbox, dpi=self.dpi
            )
            description = self.figure_descriptor.describe(crop)
            return ElementContent(description=description)

        return None

    def _extractor_for(self, region: Region) -> str:
        if region.content is not None:
            return self.segmenter.tool_name
        if region.region_type in _TEXT_TYPES:
            return self.text_extractor.tool_name
        if region.region_type == ElementType.TABLE:
            return self.table_extractor.tool_name
        if region.region_type == ElementType.FORMULA:
            return self.formula_extractor.tool_name
        return self.figure_descriptor.tool_name

    def _make_doc_id(self, pdf_path: Path) -> str:
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _make_element_id(self, doc_id: str, region: Region) -> str:
        x0, y0, x1, y1 = (round(v) for v in region.bbox)
        raw = (
            f"{doc_id}:{region.page}:{region.region_type.value}"
            f":{x0},{y0},{x1},{y1}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _assert_output_dir_clean(self) -> None:
        """Refuse to mix artefacts from different runs in the same directory."""
        content_list = self.output_dir / "content_list.json"
        segmentation = self.output_dir / "segmentation.json"
        pages_dir = self.output_dir / "pages"
        conflicts: list[str] = []
        if content_list.exists():
            conflicts.append(str(content_list))
        if segmentation.exists():
            conflicts.append(str(segmentation))
        if pages_dir.exists() and any(pages_dir.iterdir()):
            conflicts.append(str(pages_dir))
        if conflicts:
            raise FileExistsError(
                "Extraction output dir already contains artefacts: "
                + ", ".join(conflicts)
                + ". Choose a different --output directory or remove these files."
            )
