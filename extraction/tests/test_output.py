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


def test_write_content_list(tmp_path: Path) -> None:
    cl = _make_content_list()
    writer = OutputWriter(tmp_path)
    writer.write_content_list(cl)

    path = tmp_path / "content_list.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["doc_id"] == "test123"
    assert len(data["elements"]) == 1
    assert data["elements"][0]["content"]["text"] == "Hello world"


def test_write_document_rich(tmp_path: Path) -> None:
    dr = _make_document_rich()
    writer = OutputWriter(tmp_path)
    writer.write_document_rich(dr)

    path = tmp_path / "document_rich.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data["sections"]) == 1
    assert data["relations"][0]["source"] == "e001"


def test_save_page_image(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    img = Image.new("RGB", (100, 100), color="red")
    saved = writer.save_page_image(page=1, image=img)

    assert saved.exists()
    assert saved == tmp_path / "pages" / "1" / "page.png"


def test_save_element_crop(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    img = Image.new("RGB", (50, 30), color="blue")
    saved = writer.save_element_crop(
        page=2, element_id="e006", element_type="table", image=img
    )

    assert saved.exists()
    assert saved == tmp_path / "pages" / "2" / "e006_table.png"


def test_crop_from_page_image(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    page_img = Image.new("RGB", (1000, 800), color="white")
    bbox = [100.0, 200.0, 500.0, 400.0]
    crop = writer.crop_region(page_img, bbox, dpi=72)

    assert crop.size == (400, 200)


def test_content_list_excludes_none_in_content(tmp_path: Path) -> None:
    cl = _make_content_list()
    writer = OutputWriter(tmp_path)
    writer.write_content_list(cl)

    data = json.loads((tmp_path / "content_list.json").read_text())
    content = data["elements"][0]["content"]
    assert "text" in content
    assert "markdown" not in content
    assert "latex" not in content


def test_write_element_sidecar_roundtrip(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    element = Element(
        element_id="abc1234567890def",
        type=ElementType.TABLE,
        page=2,
        bbox=[10, 20, 100, 200],
        reading_order_index=3,
        confidence=0.87,
        extractor="mineru25",
        content=ElementContent(markdown="| a | b |\n|---|---|\n| 1 | 2 |", text="a b 1 2"),
    )

    path = writer.write_element_sidecar(element)

    assert path == tmp_path / "pages" / "2" / "abc1234567890def_table.json"
    assert path.exists()
    loaded = Element.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert loaded == element


def test_build_content_list_from_sidecars_preserves_order(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    # Write three elements on two pages, deliberately out of order.
    elements_in = [
        Element(
            element_id="c" * 16,
            type=ElementType.TEXT,
            page=1,
            bbox=[0, 0, 100, 50],
            reading_order_index=0,
            confidence=0.9,
            extractor="mock",
            content=ElementContent(text="page 1 text"),
        ),
        Element(
            element_id="a" * 16,
            type=ElementType.HEADING,
            page=0,
            bbox=[0, 0, 100, 50],
            reading_order_index=5,
            confidence=0.9,
            extractor="mock",
            content=ElementContent(text="heading"),
        ),
        Element(
            element_id="b" * 16,
            type=ElementType.TEXT,
            page=0,
            bbox=[0, 60, 100, 200],
            reading_order_index=7,
            confidence=0.9,
            extractor="mock",
            content=ElementContent(text="page 0 body"),
        ),
    ]
    for el in elements_in:
        writer.write_element_sidecar(el)

    cl = writer.build_content_list(
        doc_id="doc1", source_file="x.pdf", total_pages=2, segmentation_tool="mock"
    )

    # Order: (page, reading_order_index, element_id) → page 0: a(5), b(7); page 1: c(0)
    ids = [e.element_id for e in cl.elements]
    assert ids == ["a" * 16, "b" * 16, "c" * 16]
    # Reading order re-numbered globally.
    assert [e.reading_order_index for e in cl.elements] == [0, 1, 2]
    # Pages populated with the right element_ids.
    assert len(cl.pages) == 2
    assert cl.pages[0].element_ids == ["a" * 16, "b" * 16]
    assert cl.pages[1].element_ids == ["c" * 16]
    assert cl.pages[0].image_path == "pages/0/page.png"


def test_crop_region_scales_points_to_pixels(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    page_img = Image.new("RGB", (1240, 1754), color="white")  # ~A4 at 150 DPI
    bbox_points = [100.0, 200.0, 200.0, 300.0]
    crop = writer.crop_region(page_img, bbox_points, dpi=150)
    # 100 pt * 150/72 = 208.33 → int = 208 (x0); 200 pt * 150/72 = 416.67 → int(+0.999) = 417 (x1)
    # width = 417 - 208 = 209
    # 200 pt * 150/72 = 416.67 → int = 416 (y0); 300 pt * 150/72 = 625.0 → int(+0.999) = 625 (y1)
    # height = 625 - 416 = 209
    assert crop.size == (209, 209)


def test_crop_region_clamps_to_image_bounds(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    page_img = Image.new("RGB", (500, 500), color="white")
    # bbox extends beyond image in every direction (in pixel space already)
    bbox_points = [-20.0, -20.0, 10_000.0, 10_000.0]
    crop = writer.crop_region(page_img, bbox_points, dpi=72)  # scale=1
    assert crop.size == (500, 500)


def test_crop_region_default_dpi_is_72_backwards_compat(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path)
    page_img = Image.new("RGB", (1000, 800), color="white")
    # No dpi → default 72 → bbox treated as pixels
    crop = writer.crop_region(page_img, [100.0, 200.0, 500.0, 400.0], dpi=72)
    assert crop.size == (400, 200)


def test_mark_stage_done_creates_marker(tmp_path):
    writer = OutputWriter(tmp_path)
    writer.mark_stage_done("segment")
    assert (tmp_path / ".stages" / "segment.done").exists()


def test_is_stage_done_reflects_marker(tmp_path):
    writer = OutputWriter(tmp_path)
    assert writer.is_stage_done("segment") is False
    writer.mark_stage_done("segment")
    assert writer.is_stage_done("segment") is True


def test_mark_stage_done_clears_previous_error(tmp_path):
    writer = OutputWriter(tmp_path)
    writer.write_stage_error("segment", RuntimeError("boom"))
    writer.mark_stage_done("segment")
    assert (tmp_path / ".stages" / "segment.done").exists()
    assert not (tmp_path / ".stages" / "segment.error").exists()


def test_write_stage_error_writes_traceback(tmp_path):
    writer = OutputWriter(tmp_path)
    try:
        raise RuntimeError("something broke")
    except RuntimeError as exc:
        writer.write_stage_error("segment", exc)
    error_path = tmp_path / ".stages" / "segment.error"
    assert error_path.exists()
    text = error_path.read_text(encoding="utf-8")
    assert "RuntimeError" in text
    assert "something broke" in text


def test_write_stage_error_clears_previous_done(tmp_path):
    writer = OutputWriter(tmp_path)
    writer.mark_stage_done("segment")
    writer.write_stage_error("segment", RuntimeError("regression"))
    assert (tmp_path / ".stages" / "segment.error").exists()
    assert not (tmp_path / ".stages" / "segment.done").exists()
