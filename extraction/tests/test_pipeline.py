import json
from pathlib import Path

from PIL import Image

from extraction.models import ElementContent, ElementType, Region
from extraction.pipeline import ExtractionPipeline


class MockRenderer:
    tool_name = "mock_renderer"

    def page_count(self, pdf_path: Path) -> int:
        return 2

    def render_page(self, pdf_path: Path, page_number: int) -> Image.Image:
        return Image.new("RGB", (1000, 800), color="white")

    def render_all(self, pdf_path: Path) -> list[Image.Image]:
        return [self.render_page(pdf_path, i) for i in range(self.page_count(pdf_path))]


class MockSegmenter:
    tool_name = "mock_seg"

    def segment(self, pdf_path: Path) -> list[Region]:
        return [
            Region(
                page=0,
                bbox=[80, 40, 900, 90],
                region_type=ElementType.HEADING,
                confidence=0.98,
                content=ElementContent(text="1. Einleitung"),
            ),
            Region(
                page=0,
                bbox=[80, 100, 900, 300],
                region_type=ElementType.TEXT,
                confidence=0.95,
            ),
            Region(
                page=1,
                bbox=[100, 200, 800, 500],
                region_type=ElementType.TABLE,
                confidence=0.93,
                content=ElementContent(
                    markdown="| A | B |\n|---|---|\n| 1 | 2 |",
                    text="A B 1 2",
                    caption="Tabelle 1",
                ),
            ),
        ]


class MockTextExtractor:
    tool_name = "mock_ocr"

    def extract(self, page_image: Image.Image, page_number: int) -> ElementContent:
        return ElementContent(text="Extracted text from page")


class MockTableExtractor:
    tool_name = "mock_table"

    def extract(self, region_image: Image.Image, page_number: int) -> ElementContent:
        return ElementContent(markdown="| X | Y |", text="X Y")


class MockFormulaExtractor:
    tool_name = "mock_formula"

    def extract(self, region_image: Image.Image, page_number: int) -> ElementContent:
        return ElementContent(latex="E=mc^2", text="E=mc^2")


class MockFigureDescriptor:
    tool_name = "mock_fig"

    def describe(self, image: Image.Image) -> str:
        return "A test figure"


def test_pipeline_produces_output_files(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf content")
    output_dir = tmp_path / "output"

    pipeline = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=output_dir,
        confidence_threshold=0.3,
    )
    pipeline.run(pdf_path)

    assert (output_dir / "content_list.json").exists()
    assert not (output_dir / "document_rich.json").exists()
    # At least one sidecar per page from the mock segmenter
    assert list((output_dir / "pages" / "0").glob("*_heading.json"))
    assert list((output_dir / "pages" / "0").glob("*_text.json"))
    assert list((output_dir / "pages" / "1").glob("*_table.json"))


def test_pipeline_elements_in_reading_order(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake")
    output_dir = tmp_path / "output"

    pipeline = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=output_dir,
        confidence_threshold=0.3,
    )
    pipeline.run(pdf_path)

    data = json.loads((output_dir / "content_list.json").read_text())
    elements = data["elements"]
    indices = [e["reading_order_index"] for e in elements]
    assert indices == sorted(indices)
    assert elements[0]["type"] == "heading"
    assert elements[2]["type"] == "table"


def test_pipeline_page_images_saved(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake")
    output_dir = tmp_path / "output"

    pipeline = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=output_dir,
        confidence_threshold=0.3,
    )
    pipeline.run(pdf_path)

    assert (output_dir / "pages" / "0" / "page.png").exists()
    assert (output_dir / "pages" / "1" / "page.png").exists()


def test_pipeline_filters_low_confidence(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake")
    output_dir = tmp_path / "output"

    pipeline = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=output_dir,
        confidence_threshold=0.99,
    )
    pipeline.run(pdf_path)

    data = json.loads((output_dir / "content_list.json").read_text())
    assert len(data["elements"]) == 0


def test_pipeline_text_regions_get_extracted(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake")
    output_dir = tmp_path / "output"

    pipeline = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=output_dir,
        confidence_threshold=0.3,
    )
    pipeline.run(pdf_path)

    data = json.loads((output_dir / "content_list.json").read_text())
    text_elements = [e for e in data["elements"] if e["type"] == "text"]
    assert len(text_elements) == 1
    assert text_elements[0]["content"]["text"] == "Extracted text from page"
