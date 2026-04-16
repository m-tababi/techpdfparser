import json
from pathlib import Path

from PIL import Image

from extraction.models import (
    ContentList,
    DocumentRich,
    Element,
    ElementContent,
    ElementType,
    PageInfo,
    Relation,
    Section,
)
from extraction.output import OutputWriter


def _make_content_list() -> ContentList:
    return ContentList(
        doc_id="test123",
        source_file="test.pdf",
        total_pages=1,
        segmentation_tool="mineru25",
        pages=[PageInfo(page=1, image_path="pages/1/page.png", element_ids=["e001"])],
        elements=[
            Element(
                element_id="e001",
                type=ElementType.TEXT,
                page=1,
                bbox=[0, 0, 100, 50],
                reading_order_index=0,
                section_path=["1. Intro"],
                confidence=0.95,
                extractor="olmocr2",
                content=ElementContent(text="Hello world"),
            )
        ],
    )


def _make_document_rich() -> DocumentRich:
    return DocumentRich(
        doc_id="test123",
        source_file="test.pdf",
        total_pages=1,
        segmentation_tool="mineru25",
        sections=[
            Section(heading="1. Intro", level=1, page_start=1, children=["e001"])
        ],
        relations=[
            Relation(source="e001", target="e002", type="refers_to", evidence="see table")
        ],
    )


def test_write_content_list(tmp_path: Path):
    cl = _make_content_list()
    writer = OutputWriter(tmp_path)
    writer.write_content_list(cl)

    path = tmp_path / "content_list.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["doc_id"] == "test123"
    assert len(data["elements"]) == 1
    assert data["elements"][0]["content"]["text"] == "Hello world"


def test_write_document_rich(tmp_path: Path):
    dr = _make_document_rich()
    writer = OutputWriter(tmp_path)
    writer.write_document_rich(dr)

    path = tmp_path / "document_rich.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data["sections"]) == 1
    assert data["relations"][0]["source"] == "e001"


def test_save_page_image(tmp_path: Path):
    writer = OutputWriter(tmp_path)
    img = Image.new("RGB", (100, 100), color="red")
    saved = writer.save_page_image(page=1, image=img)

    assert saved.exists()
    assert saved == tmp_path / "pages" / "1" / "page.png"


def test_save_element_crop(tmp_path: Path):
    writer = OutputWriter(tmp_path)
    img = Image.new("RGB", (50, 30), color="blue")
    saved = writer.save_element_crop(
        page=2, element_id="e006", element_type="table", image=img
    )

    assert saved.exists()
    assert saved == tmp_path / "pages" / "2" / "e006_table.png"


def test_crop_from_page_image(tmp_path: Path):
    writer = OutputWriter(tmp_path)
    page_img = Image.new("RGB", (1000, 800), color="white")
    bbox = [100, 200, 500, 400]
    crop = writer.crop_region(page_img, bbox)

    assert crop.size == (400, 200)


def test_content_list_excludes_none_in_content(tmp_path: Path):
    cl = _make_content_list()
    writer = OutputWriter(tmp_path)
    writer.write_content_list(cl)

    data = json.loads((tmp_path / "content_list.json").read_text())
    content = data["elements"][0]["content"]
    assert "text" in content
    assert "markdown" not in content
    assert "latex" not in content
