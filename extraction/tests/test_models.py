
from extraction.models import (
    ContentList,
    DocumentRich,
    Element,
    ElementContent,
    ElementType,
    PageInfo,
    Region,
    Relation,
    Section,
)


def test_element_serializes_to_spec_format():
    el = Element(
        element_id="e001",
        type=ElementType.HEADING,
        page=1,
        bbox=[80, 40, 900, 90],
        reading_order_index=0,
        section_path=["1. Einleitung"],
        confidence=0.98,
        extractor="olmocr2",
        content=ElementContent(text="1. Einleitung"),
    )
    data = el.model_dump(mode="json")
    assert data["element_id"] == "e001"
    assert data["type"] == "heading"
    assert data["bbox"] == [80, 40, 900, 90]
    assert data["content"]["text"] == "1. Einleitung"
    assert data["extractor"] == "olmocr2"


def test_table_element_has_all_content_fields():
    el = Element(
        element_id="e006",
        type=ElementType.TABLE,
        page=2,
        bbox=[100, 220, 880, 450],
        reading_order_index=5,
        section_path=["2. Messergebnisse"],
        confidence=0.93,
        extractor="mineru25",
        content=ElementContent(
            markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            text="A B 1 2",
            image_path="pages/2/e006_table.png",
            caption="Tabelle 1: Messwerte",
        ),
    )
    data = el.model_dump(mode="json", exclude_none=True)
    assert data["content"]["markdown"].startswith("| A")
    assert data["content"]["caption"] == "Tabelle 1: Messwerte"
    assert data["content"]["image_path"] == "pages/2/e006_table.png"


def test_content_list_round_trips_json():
    cl = ContentList(
        doc_id="abc123",
        source_file="test.pdf",
        total_pages=1,
        schema_version="1.0",
        segmentation_tool="mineru25",
        pages=[PageInfo(page=1, image_path="pages/1/page.png", element_ids=["e001"])],
        elements=[
            Element(
                element_id="e001",
                type=ElementType.TEXT,
                page=1,
                bbox=[0, 0, 100, 100],
                reading_order_index=0,
                section_path=[],
                confidence=0.9,
                extractor="olmocr2",
                content=ElementContent(text="Hello"),
            )
        ],
    )
    json_str = cl.model_dump_json(indent=2)
    parsed = ContentList.model_validate_json(json_str)
    assert parsed.doc_id == "abc123"
    assert len(parsed.elements) == 1
    assert parsed.elements[0].content.text == "Hello"


def test_document_rich_sections_and_relations():
    dr = DocumentRich(
        doc_id="abc123",
        source_file="test.pdf",
        total_pages=2,
        schema_version="1.0",
        segmentation_tool="mineru25",
        sections=[
            Section(
                heading="1. Einleitung",
                level=1,
                page_start=1,
                children=["e001", "e002"],
            )
        ],
        relations=[
            Relation(
                source="e001",
                target="e002",
                type="refers_to",
                evidence="siehe Tabelle 1",
            )
        ],
    )
    data = dr.model_dump(mode="json")
    assert data["sections"][0]["children"] == ["e001", "e002"]
    assert data["relations"][0]["source"] == "e001"


def test_region_holds_segmentation_data():
    r = Region(
        page=1,
        bbox=[100, 200, 500, 400],
        region_type=ElementType.TABLE,
        confidence=0.95,
    )
    assert r.region_type == ElementType.TABLE
    assert r.content is None


def test_element_content_excludes_none_fields():
    c = ElementContent(text="just text")
    data = c.model_dump(exclude_none=True)
    assert "text" in data
    assert "markdown" not in data
    assert "latex" not in data
    assert "image_path" not in data
