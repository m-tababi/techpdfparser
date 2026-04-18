import json
from pathlib import Path

import pytest
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


def test_pipeline_produces_output_files(tmp_path: Path) -> None:
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


def test_pipeline_elements_in_reading_order(tmp_path: Path) -> None:
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


def test_pipeline_page_images_saved(tmp_path: Path) -> None:
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


def test_pipeline_filters_low_confidence(tmp_path: Path) -> None:
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


def test_pipeline_text_regions_get_extracted(tmp_path: Path) -> None:
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


def test_pipeline_source_file_is_filename_only(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "dir"
    nested.mkdir(parents=True)
    pdf_path = nested / "report.pdf"
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
    assert data["source_file"] == "report.pdf"


def test_pipeline_stores_dpi_default_and_override(tmp_path: Path) -> None:
    default = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=tmp_path / "out_default",
        confidence_threshold=0.3,
    )
    overridden = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=tmp_path / "out_300",
        confidence_threshold=0.3,
        dpi=300,
    )
    assert default.dpi == 150
    assert overridden.dpi == 300


def test_element_id_is_path_independent(tmp_path: Path) -> None:
    pdf_a = tmp_path / "a" / "doc.pdf"
    pdf_b = tmp_path / "b" / "doc.pdf"
    pdf_a.parent.mkdir(parents=True)
    pdf_b.parent.mkdir(parents=True)
    pdf_a.write_bytes(b"identical content")
    pdf_b.write_bytes(b"identical content")

    def run_one(pdf: Path, out: Path) -> list[str]:
        ExtractionPipeline(
            renderer=MockRenderer(),
            segmenter=MockSegmenter(),
            text_extractor=MockTextExtractor(),
            table_extractor=MockTableExtractor(),
            formula_extractor=MockFormulaExtractor(),
            figure_descriptor=MockFigureDescriptor(),
            output_dir=out,
            confidence_threshold=0.3,
        ).run(pdf)
        data = json.loads((out / "content_list.json").read_text())
        return [e["element_id"] for e in data["elements"]]

    ids_a = run_one(pdf_a, tmp_path / "out_a")
    ids_b = run_one(pdf_b, tmp_path / "out_b")
    assert ids_a == ids_b


def test_element_id_differs_on_region_type_at_same_bbox(tmp_path: Path) -> None:
    from extraction.models import Region
    from extraction.pipeline import ExtractionPipeline as _P
    pipe = _P(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=tmp_path,
        confidence_threshold=0.3,
    )
    r_text = Region(page=0, bbox=[10, 20, 30, 40], region_type=ElementType.TEXT, confidence=1.0)
    r_head = Region(page=0, bbox=[10, 20, 30, 40], region_type=ElementType.HEADING, confidence=1.0)
    id1 = pipe._make_element_id("docid", r_text)
    id2 = pipe._make_element_id("docid", r_head)
    assert id1 != id2


def _make_pipeline(output_dir: Path) -> ExtractionPipeline:
    return ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=MockSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=output_dir,
        confidence_threshold=0.3,
    )


def test_pipeline_aborts_when_content_list_exists(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()
    (out / "content_list.json").write_text("{}")

    with pytest.raises(FileExistsError):
        _make_pipeline(out).run(pdf_path)


def test_pipeline_aborts_when_segmentation_json_exists(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()
    (out / "segmentation.json").write_text("[]")

    with pytest.raises(FileExistsError):
        _make_pipeline(out).run(pdf_path)


def test_pipeline_aborts_when_pages_dir_is_nonempty(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    (out / "pages" / "0").mkdir(parents=True)
    (out / "pages" / "0" / "leftover.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(FileExistsError):
        _make_pipeline(out).run(pdf_path)


def test_pipeline_allows_empty_output_dir(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()
    _make_pipeline(out).run(pdf_path)  # must not raise
    assert (out / "content_list.json").exists()


class EmptyTextSegmenter:
    """Emits text regions with empty content and a figure with only bbox."""
    tool_name = "empty_seg"

    def segment(self, pdf_path: Path) -> list[Region]:
        return [
            Region(
                page=0, bbox=[0, 0, 100, 50],
                region_type=ElementType.TEXT, confidence=0.9,
                content=ElementContent(text="   "),
            ),
            Region(
                page=0, bbox=[0, 60, 100, 200],
                region_type=ElementType.TEXT, confidence=0.9,
                content=ElementContent(text=""),
            ),
            Region(
                page=0, bbox=[0, 220, 100, 300],
                region_type=ElementType.FIGURE, confidence=0.9,
                content=None,
            ),
        ]


class EmptyTableSegmenter:
    tool_name = "empty_seg"

    def segment(self, pdf_path: Path) -> list[Region]:
        return [
            Region(
                page=0, bbox=[0, 0, 100, 50],
                region_type=ElementType.TABLE, confidence=0.9,
                content=ElementContent(),
            ),
        ]


def test_pipeline_drops_empty_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"
    pipe = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=EmptyTextSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=MockTableExtractor(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=out,
        confidence_threshold=0.3,
    )
    pipe.run(pdf_path)
    data = json.loads((out / "content_list.json").read_text())
    types = sorted(e["type"] for e in data["elements"])
    # Under the current (pre-Task 8) merge rule the segmenter's empty content
    # flows through unchanged. _is_droppable must remove both empty-text
    # regions; the figure survives because the descriptor gives it a non-empty
    # description. Task 8 Step 5 revisits this test when the merge rule flips.
    assert types == ["figure"]


def test_pipeline_drops_table_without_markdown_or_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake")
    out = tmp_path / "out"

    class NoopTable:
        tool_name = "empty_seg"
        def extract(self, region_image, page_number):
            return ElementContent()

    pipe = ExtractionPipeline(
        renderer=MockRenderer(),
        segmenter=EmptyTableSegmenter(),
        text_extractor=MockTextExtractor(),
        table_extractor=NoopTable(),
        formula_extractor=MockFormulaExtractor(),
        figure_descriptor=MockFigureDescriptor(),
        output_dir=out,
        confidence_threshold=0.3,
    )
    pipe.run(pdf_path)
    data = json.loads((out / "content_list.json").read_text())
    assert not any(e["type"] == "table" for e in data["elements"])
