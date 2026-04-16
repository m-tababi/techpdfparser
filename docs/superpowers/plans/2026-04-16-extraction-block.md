# Extraction Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent extraction block under `extraction/` that takes a PDF and produces `content_list.json` + `document_rich.json` + image crops, with no embedding or indexing.

**Architecture:** Segmentation-first pipeline: a segmenter analyzes PDF layout into typed regions, each region is routed to the best extractor, results are merged in reading order with confidence filtering, and written to the spec output format. All tools are swappable via config and registry.

**Tech Stack:** Python 3.10+, Pydantic v2, PyYAML, Pillow, PyMuPDF (fitz), pytest

**Spec:** `docs/superpowers/specs/2026-04-16-extraction-block-design.md`

---

## File Structure

```
extraction/
├── __init__.py              # Package marker
├── __main__.py              # CLI: python -m extraction extract path.pdf
├── config.py                # ExtractionConfig Pydantic model + YAML loading
├── interfaces.py            # Protocol classes for all swappable components
├── models.py                # Element, ElementContent, PageInfo, ContentList, DocumentRich, Region
├── output.py                # Write content_list.json, document_rich.json, manage image crops
├── pipeline.py              # Orchestration: segment -> route -> extract -> merge -> write
├── registry.py              # Adapter registration/lookup (same pattern as src/core/registry.py)
├── adapters/
│   ├── __init__.py          # Import all adapters to trigger registration
│   └── pymupdf_renderer.py  # Page rendering + bbox cropping
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_output.py
    ├── test_registry.py
    ├── test_config.py
    └── test_pipeline.py
```

Notes:
- ML-heavy adapters (MinerU, OlmOCR, Qwen2.5-VL, PPFormulaNet) are NOT part of this plan. They are ported separately after the framework is validated.
- This plan builds the complete framework + one real adapter (PyMuPDF renderer) + mock-adapter-based pipeline tests.
- The pipeline is fully testable with mock adapters that return fixture data.

---

### Task 1: Scaffolding

**Files:**
- Create: `extraction/__init__.py`
- Create: `extraction/tests/__init__.py`
- Create: `extraction/adapters/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p extraction/adapters extraction/tests
```

- [ ] **Step 2: Create package markers**

`extraction/__init__.py`:
```python
"""Extraction block — PDF to structured output, no embedding."""
```

`extraction/tests/__init__.py`:
```python
```

`extraction/adapters/__init__.py`:
```python
"""Import all adapter modules to trigger @register_* decorators."""
from . import pymupdf_renderer  # noqa: F401
```

- [ ] **Step 3: Update pyproject.toml — add extraction to mypy and test paths**

Add `"extraction"` to the mypy `files` list and add `"extraction/tests"` to pytest `testpaths`:

```toml
[tool.mypy]
files = [
    "src/__main__.py",
    "src/core",
    "src/utils",
    "src/adapters/chunkers",
    "src/adapters/vectordb",
    "extraction",
]

[tool.pytest.ini_options]
testpaths = ["tests", "extraction/tests"]
```

- [ ] **Step 4: Verify structure**

Run: `python -c "import extraction; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add extraction/ pyproject.toml
git commit -m "feat(extraction): scaffold extraction block directory structure"
```

---

### Task 2: Data Models

**Files:**
- Create: `extraction/models.py`
- Create: `extraction/tests/test_models.py`

The models define the output schema from the spec: elements with typed content, page info, content list, document structure, and segmentation regions.

- [ ] **Step 1: Write the failing test**

`extraction/tests/test_models.py`:
```python
import json

from extraction.models import (
    ContentList,
    DocumentRich,
    Element,
    ElementContent,
    ElementType,
    PageInfo,
    Region,
    Section,
    Relation,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest extraction/tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'extraction.models'`

- [ ] **Step 3: Write the models**

`extraction/models.py`:
```python
"""Data models for the extraction block output format.

Defines the schema for content_list.json, document_rich.json,
and the intermediate segmentation regions.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FORMULA = "formula"
    FIGURE = "figure"
    DIAGRAM = "diagram"
    TECHNICAL_DRAWING = "technical_drawing"


class ElementContent(BaseModel):
    """Content fields vary by element type. All are optional; presence depends on type."""

    text: str | None = None
    markdown: str | None = None
    latex: str | None = None
    image_path: str | None = None
    description: str | None = None
    caption: str | None = None


class Element(BaseModel):
    """One extractable unit from the PDF — a text block, table, formula, figure, etc."""

    element_id: str
    type: ElementType
    page: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    reading_order_index: int
    section_path: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    extractor: str
    content: ElementContent


class PageInfo(BaseModel):
    """Per-page metadata linking the page image to its elements."""

    page: int
    image_path: str
    element_ids: list[str] = Field(default_factory=list)


class ContentList(BaseModel):
    """Root model for content_list.json — flat element list in reading order."""

    doc_id: str
    source_file: str
    total_pages: int
    schema_version: str = "1.0"
    segmentation_tool: str
    pages: list[PageInfo] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)


class Section(BaseModel):
    """A section in the document hierarchy."""

    heading: str
    level: int
    page_start: int
    children: list[str] = Field(default_factory=list)
    subsections: list[Section] = Field(default_factory=list)


class Relation(BaseModel):
    """A reference between two elements (e.g. text refers to a table)."""

    source: str
    target: str
    type: str
    evidence: str = ""


class DocumentRich(BaseModel):
    """Root model for document_rich.json — hierarchical structure + relations."""

    doc_id: str
    source_file: str
    total_pages: int
    schema_version: str = "1.0"
    segmentation_tool: str
    sections: list[Section] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class Region(BaseModel):
    """A detected region from layout segmentation, before extraction."""

    page: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    region_type: ElementType
    confidence: float = Field(ge=0.0, le=1.0)
    content: ElementContent | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest extraction/tests/test_models.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add extraction/models.py extraction/tests/test_models.py
git commit -m "feat(extraction): add data models for output format"
```

---

### Task 3: Output Writer

**Files:**
- Create: `extraction/output.py`
- Create: `extraction/tests/test_output.py`

Writes `content_list.json` and `document_rich.json` from model objects. Handles image crop saving.

- [ ] **Step 1: Write the failing tests**

`extraction/tests/test_output.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest extraction/tests/test_output.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'extraction.output'`

- [ ] **Step 3: Write the output writer**

`extraction/output.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest extraction/tests/test_output.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add extraction/output.py extraction/tests/test_output.py
git commit -m "feat(extraction): add output writer for content_list and document_rich"
```

---

### Task 4: Interfaces + Registry

**Files:**
- Create: `extraction/interfaces.py`
- Create: `extraction/registry.py`
- Create: `extraction/tests/test_registry.py`

Defines Protocol classes for all swappable components and the registry pattern for config-driven adapter lookup.

- [ ] **Step 1: Write the failing test**

`extraction/tests/test_registry.py`:
```python
from extraction.registry import (
    get_renderer,
    get_segmenter,
    get_text_extractor,
    get_table_extractor,
    get_formula_extractor,
    get_figure_descriptor,
    register_renderer,
    register_segmenter,
    register_text_extractor,
    register_table_extractor,
    register_formula_extractor,
    register_figure_descriptor,
)


def test_register_and_get_renderer():
    @register_renderer("test_renderer")
    class TestRenderer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    instance = get_renderer("test_renderer", dpi=150)
    assert instance.kwargs == {"dpi": 150}


def test_register_and_get_segmenter():
    @register_segmenter("test_seg")
    class TestSeg:
        pass

    instance = get_segmenter("test_seg")
    assert instance is not None


def test_unknown_adapter_raises_key_error():
    import pytest

    with pytest.raises(KeyError, match="Unknown renderer"):
        get_renderer("nonexistent_adapter_xyz")


def test_register_all_adapter_types():
    @register_text_extractor("test_ocr")
    class T1:
        pass

    @register_table_extractor("test_table")
    class T2:
        pass

    @register_formula_extractor("test_formula")
    class T3:
        pass

    @register_figure_descriptor("test_fig")
    class T4:
        pass

    assert get_text_extractor("test_ocr") is not None
    assert get_table_extractor("test_table") is not None
    assert get_formula_extractor("test_formula") is not None
    assert get_figure_descriptor("test_fig") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest extraction/tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'extraction.registry'`

- [ ] **Step 3: Write the interfaces**

`extraction/interfaces.py`:
```python
"""Protocol classes for all swappable extraction components.

Each adapter type has a Protocol that defines the interface.
Concrete adapters register via decorators in extraction/registry.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL.Image import Image

from .models import ElementContent, Region


class PageRenderer(Protocol):
    @property
    def tool_name(self) -> str: ...

    def page_count(self, pdf_path: Path) -> int: ...

    def render_page(self, pdf_path: Path, page_number: int) -> Image: ...

    def render_all(self, pdf_path: Path) -> list[Image]: ...


class Segmenter(Protocol):
    @property
    def tool_name(self) -> str: ...

    def segment(self, pdf_path: Path) -> list[Region]:
        """Analyze layout and return typed regions for all pages."""
        ...


class TextExtractor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def extract(self, page_image: Image, page_number: int) -> ElementContent:
        """Extract text from a page image. Returns content with text field set."""
        ...


class TableExtractor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        """Extract table content from a cropped region. Returns markdown + text."""
        ...


class FormulaExtractor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        """Extract formula from a cropped region. Returns latex + text."""
        ...


class FigureDescriptor(Protocol):
    @property
    def tool_name(self) -> str: ...

    def describe(self, image: Image) -> str:
        """Generate a text description of a figure/diagram image."""
        ...
```

- [ ] **Step 4: Write the registry**

`extraction/registry.py`:
```python
"""Adapter registry for config-driven tool switching.

Same pattern as src/core/registry.py — decorators register adapters
by name, get_* functions instantiate them.
"""
from __future__ import annotations

from typing import Any, Callable

AdapterRegistry = dict[str, type[Any]]
RegisterDecorator = Callable[[type[Any]], type[Any]]


def _make_register(registry: AdapterRegistry) -> Callable[[str], RegisterDecorator]:
    def register(name: str) -> RegisterDecorator:
        def decorator(cls: type[Any]) -> type[Any]:
            registry[name] = cls
            return cls
        return decorator
    return register


def _make_get(registry: AdapterRegistry, category: str) -> Callable[..., Any]:
    def get(name: str, **kwargs: Any) -> Any:
        if name not in registry:
            available = sorted(registry.keys())
            raise KeyError(
                f"Unknown {category} adapter '{name}'. Available: {available}"
            )
        return registry[name](**kwargs)
    return get


_RENDERERS: AdapterRegistry = {}
_SEGMENTERS: AdapterRegistry = {}
_TEXT_EXTRACTORS: AdapterRegistry = {}
_TABLE_EXTRACTORS: AdapterRegistry = {}
_FORMULA_EXTRACTORS: AdapterRegistry = {}
_FIGURE_DESCRIPTORS: AdapterRegistry = {}

register_renderer = _make_register(_RENDERERS)
register_segmenter = _make_register(_SEGMENTERS)
register_text_extractor = _make_register(_TEXT_EXTRACTORS)
register_table_extractor = _make_register(_TABLE_EXTRACTORS)
register_formula_extractor = _make_register(_FORMULA_EXTRACTORS)
register_figure_descriptor = _make_register(_FIGURE_DESCRIPTORS)

get_renderer = _make_get(_RENDERERS, "renderer")
get_segmenter = _make_get(_SEGMENTERS, "segmenter")
get_text_extractor = _make_get(_TEXT_EXTRACTORS, "text_extractor")
get_table_extractor = _make_get(_TABLE_EXTRACTORS, "table_extractor")
get_formula_extractor = _make_get(_FORMULA_EXTRACTORS, "formula_extractor")
get_figure_descriptor = _make_get(_FIGURE_DESCRIPTORS, "figure_descriptor")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest extraction/tests/test_registry.py -v`
Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add extraction/interfaces.py extraction/registry.py extraction/tests/test_registry.py
git commit -m "feat(extraction): add adapter interfaces and registry"
```

---

### Task 5: Config

**Files:**
- Create: `extraction/config.py`
- Create: `extraction/tests/test_config.py`

Config for the extraction block. Determines which adapter is used for each role.

- [ ] **Step 1: Write the failing test**

`extraction/tests/test_config.py`:
```python
from pathlib import Path

from extraction.config import ExtractionConfig, load_extraction_config


def test_default_config_has_all_fields():
    cfg = ExtractionConfig()
    assert cfg.renderer == "pymupdf"
    assert cfg.segmenter == "mineru25"
    assert cfg.text_extractor == "olmocr2"
    assert cfg.table_extractor == "mineru25"
    assert cfg.formula_extractor == "ppformulanet"
    assert cfg.figure_descriptor == "qwen25vl"
    assert cfg.output_dir == "outputs"
    assert cfg.confidence_threshold == 0.3
    assert cfg.dpi == 150


def test_load_from_yaml(tmp_path: Path):
    yaml_content = """
extraction:
  renderer: pymupdf
  segmenter: mineru25
  text_extractor: olmocr2
  output_dir: my_outputs
  confidence_threshold: 0.5

adapters:
  pymupdf:
    dpi: 300
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    cfg = load_extraction_config(config_path)

    assert cfg.output_dir == "my_outputs"
    assert cfg.confidence_threshold == 0.5
    assert cfg.adapters["pymupdf"]["dpi"] == 300


def test_get_adapter_config_returns_empty_for_unknown():
    cfg = ExtractionConfig()
    assert cfg.get_adapter_config("nonexistent") == {}


def test_get_adapter_config_returns_settings():
    cfg = ExtractionConfig(adapters={"pymupdf": {"dpi": 300}})
    assert cfg.get_adapter_config("pymupdf") == {"dpi": 300}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest extraction/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'extraction.config'`

- [ ] **Step 3: Write the config**

`extraction/config.py`:
```python
"""Extraction block configuration.

Load from YAML or use defaults. Each adapter has its own config section.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ExtractionConfig(BaseModel):
    """Configuration for the extraction pipeline."""

    renderer: str = "pymupdf"
    segmenter: str = "mineru25"
    text_extractor: str = "olmocr2"
    table_extractor: str = "mineru25"
    formula_extractor: str = "ppformulanet"
    figure_descriptor: str = "qwen25vl"
    output_dir: str = "outputs"
    confidence_threshold: float = 0.3
    dpi: int = 150
    adapters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_adapter_config(self, adapter_name: str) -> dict[str, Any]:
        return self.adapters.get(adapter_name, {})


def load_extraction_config(path: str | Path) -> ExtractionConfig:
    """Load extraction config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    extraction_raw = raw.get("extraction", {})
    extraction_raw["adapters"] = raw.get("adapters", {})
    return ExtractionConfig.model_validate(extraction_raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest extraction/tests/test_config.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add extraction/config.py extraction/tests/test_config.py
git commit -m "feat(extraction): add extraction config with YAML loading"
```

---

### Task 6: Page Renderer Adapter (PyMuPDF)

**Files:**
- Create: `extraction/adapters/pymupdf_renderer.py`
- Modify: `extraction/adapters/__init__.py` (already done in Task 1)

This is the one real adapter in this plan. Renders PDF pages to PIL Images and crops regions by bounding box. Ported from `src/adapters/renderers/pymupdf.py`.

- [ ] **Step 1: Write the adapter**

`extraction/adapters/pymupdf_renderer.py`:
```python
"""Page renderer using PyMuPDF (fitz).

Ported from src/adapters/renderers/pymupdf.py for the independent
extraction block. Renders PDF pages to PIL Images.
"""
from __future__ import annotations

from pathlib import Path

import PIL.Image

from ..registry import register_renderer


@register_renderer("pymupdf")
class PyMuPDFRenderer:
    TOOL_NAME = "pymupdf"

    def __init__(self, dpi: int = 150) -> None:
        self._dpi = dpi
        self._fitz = self._import_fitz()

    @staticmethod
    def _import_fitz():
        try:
            import fitz
            return fitz
        except ImportError:
            raise ImportError("pymupdf not installed. Run: pip install pymupdf")

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def page_count(self, pdf_path: Path) -> int:
        with self._fitz.open(str(pdf_path)) as doc:
            return doc.page_count

    def render_page(self, pdf_path: Path, page_number: int) -> PIL.Image.Image:
        with self._fitz.open(str(pdf_path)) as doc:
            page = doc[page_number]
            mat = self._fitz.Matrix(self._dpi / 72, self._dpi / 72)
            pixmap = page.get_pixmap(matrix=mat, alpha=False)
            return PIL.Image.frombytes(
                "RGB", [pixmap.width, pixmap.height], pixmap.samples
            )

    def render_all(self, pdf_path: Path) -> list[PIL.Image.Image]:
        count = self.page_count(pdf_path)
        return [self.render_page(pdf_path, i) for i in range(count)]
```

- [ ] **Step 2: Verify adapter registers**

Run: `python -c "import extraction.adapters; from extraction.registry import get_renderer; r = get_renderer('pymupdf'); print(r.tool_name)"`
Expected: `pymupdf`

- [ ] **Step 3: Commit**

```bash
git add extraction/adapters/pymupdf_renderer.py
git commit -m "feat(extraction): add PyMuPDF renderer adapter"
```

---

### Task 7: Pipeline Orchestration

**Files:**
- Create: `extraction/pipeline.py`
- Create: `extraction/tests/test_pipeline.py`

The core orchestration: takes a PDF path and config, runs segmentation, routes regions to extractors, merges results, writes output. Tested entirely with mock adapters.

- [ ] **Step 1: Write the failing test**

`extraction/tests/test_pipeline.py`:
```python
import json
from pathlib import Path

from PIL import Image

from extraction.config import ExtractionConfig
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
    # Create a dummy PDF file (pipeline uses renderer mock, not actual PDF)
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
    assert (output_dir / "document_rich.json").exists()


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

    # Elements should be sorted by reading_order_index
    indices = [e["reading_order_index"] for e in elements]
    assert indices == sorted(indices)

    # Heading should come first, then text, then table
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

    # Page images for both pages
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
        confidence_threshold=0.99,  # Very high threshold
    )
    pipeline.run(pdf_path)

    data = json.loads((output_dir / "content_list.json").read_text())
    # Only the heading (0.98) meets the threshold — but even that is below 0.99
    # All elements should be filtered out
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
    # The text region had no pre-extracted content, so the TextExtractor ran
    assert len(text_elements) == 1
    assert text_elements[0]["content"]["text"] == "Extracted text from page"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest extraction/tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'extraction.pipeline'`

- [ ] **Step 3: Write the pipeline**

`extraction/pipeline.py`:
```python
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
    DocumentRich,
    Element,
    ElementContent,
    ElementType,
    PageInfo,
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
    ) -> None:
        self.renderer = renderer
        self.segmenter = segmenter
        self.text_extractor = text_extractor
        self.table_extractor = table_extractor
        self.formula_extractor = formula_extractor
        self.figure_descriptor = figure_descriptor
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold

    def run(self, pdf_path: Path) -> ContentList:
        """Run the full extraction pipeline on a single PDF."""
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

        # 3. Route and extract
        elements: list[Element] = []
        for idx, region in enumerate(regions):
            content = self._extract_region(region, page_images)
            if content is None:
                continue

            element_id = self._make_element_id(pdf_path, region, idx)
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
                crop = writer.crop_region(page_images[el.page], el.bbox)
                rel_path = writer.save_element_crop(
                    page=el.page,
                    element_id=el.element_id,
                    element_type=el.type.value,
                    image=crop,
                )
                el.content.image_path = str(
                    rel_path.relative_to(self.output_dir)
                )

        # 6. Reassign reading order after filtering
        for idx, el in enumerate(elements):
            el.reading_order_index = idx

        # 7. Build output models
        pages = self._build_page_infos(page_count, elements)
        content_list = ContentList(
            doc_id=self._make_doc_id(pdf_path),
            source_file=str(pdf_path),
            total_pages=page_count,
            segmentation_tool=self.segmenter.tool_name,
            pages=pages,
            elements=elements,
        )
        document_rich = DocumentRich(
            doc_id=content_list.doc_id,
            source_file=str(pdf_path),
            total_pages=page_count,
            segmentation_tool=self.segmenter.tool_name,
        )

        # 8. Write output
        writer.write_content_list(content_list)
        writer.write_document_rich(document_rich)

        return content_list

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
            crop = OutputWriter(self.output_dir).crop_region(page_img, region.bbox)
            return self.table_extractor.extract(crop, region.page)
        elif region.region_type == ElementType.FORMULA:
            crop = OutputWriter(self.output_dir).crop_region(page_img, region.bbox)
            return self.formula_extractor.extract(crop, region.page)
        elif region.region_type in {
            ElementType.FIGURE,
            ElementType.DIAGRAM,
            ElementType.TECHNICAL_DRAWING,
        }:
            crop = OutputWriter(self.output_dir).crop_region(page_img, region.bbox)
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

    def _build_page_infos(
        self, page_count: int, elements: list[Element]
    ) -> list[PageInfo]:
        pages: list[PageInfo] = []
        for p in range(page_count):
            page_elements = [e.element_id for e in elements if e.page == p]
            pages.append(
                PageInfo(
                    page=p,
                    image_path=f"pages/{p}/page.png",
                    element_ids=page_elements,
                )
            )
        return pages

    def _make_doc_id(self, pdf_path: Path) -> str:
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _make_element_id(self, pdf_path: Path, region: Region, seq: int) -> str:
        raw = f"{pdf_path}:{region.page}:{region.region_type.value}:{seq}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest extraction/tests/test_pipeline.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add extraction/pipeline.py extraction/tests/test_pipeline.py
git commit -m "feat(extraction): add pipeline orchestration with routing and merge"
```

---

### Task 8: CLI Entry Point

**Files:**
- Create: `extraction/__main__.py`
- Create: `extraction/tests/test_cli.py`

Entry point: `python -m extraction extract path/to/file.pdf --config config.yaml`

- [ ] **Step 1: Write the failing test**

`extraction/tests/test_cli.py`:
```python
import subprocess
import sys


def test_cli_shows_usage_without_args():
    result = subprocess.run(
        [sys.executable, "-m", "extraction"],
        capture_output=True,
        text=True,
    )
    assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()


def test_cli_extract_requires_pdf_arg():
    result = subprocess.run(
        [sys.executable, "-m", "extraction", "extract"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest extraction/tests/test_cli.py -v`
Expected: FAIL (no `__main__.py` yet, or module not runnable)

- [ ] **Step 3: Write the CLI**

`extraction/__main__.py`:
```python
"""CLI entrypoint: python -m extraction extract <pdf> [--config config.yaml]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ExtractionConfig, load_extraction_config
from .pipeline import ExtractionPipeline
from .registry import (
    get_figure_descriptor,
    get_formula_extractor,
    get_renderer,
    get_segmenter,
    get_table_extractor,
    get_text_extractor,
)

import extraction.adapters  # noqa: F401 — trigger adapter registration


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="extraction")
    sub = parser.add_subparsers(dest="command")

    extract = sub.add_parser("extract", help="Extract structured content from a PDF")
    extract.add_argument("pdf", type=Path, help="Path to the PDF file")
    extract.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    extract.add_argument("--output", type=Path, default=None, help="Output directory")

    return parser.parse_args()


def _load_cfg(config_path: Path | None) -> ExtractionConfig:
    if config_path is not None:
        return load_extraction_config(config_path)
    default = Path("extraction_config.yaml")
    if default.exists():
        return load_extraction_config(default)
    return ExtractionConfig()


def _run_extract(pdf_path: Path, cfg: ExtractionConfig, output_dir: Path | None) -> None:
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    out = output_dir or Path(cfg.output_dir)

    renderer = get_renderer(cfg.renderer, **cfg.get_adapter_config(cfg.renderer))
    segmenter = get_segmenter(cfg.segmenter, **cfg.get_adapter_config(cfg.segmenter))
    text_extractor = get_text_extractor(
        cfg.text_extractor, **cfg.get_adapter_config(cfg.text_extractor)
    )
    table_extractor = get_table_extractor(
        cfg.table_extractor, **cfg.get_adapter_config(cfg.table_extractor)
    )
    formula_extractor = get_formula_extractor(
        cfg.formula_extractor, **cfg.get_adapter_config(cfg.formula_extractor)
    )
    figure_descriptor = get_figure_descriptor(
        cfg.figure_descriptor, **cfg.get_adapter_config(cfg.figure_descriptor)
    )

    pipeline = ExtractionPipeline(
        renderer=renderer,
        segmenter=segmenter,
        text_extractor=text_extractor,
        table_extractor=table_extractor,
        formula_extractor=formula_extractor,
        figure_descriptor=figure_descriptor,
        output_dir=out,
        confidence_threshold=cfg.confidence_threshold,
    )

    print(f"Extracting {pdf_path.name}...")
    content_list = pipeline.run(pdf_path)
    print(f"  Elements: {len(content_list.elements)}")
    print(f"  Pages:    {content_list.total_pages}")
    print(f"  Output:   {out}")


def main() -> None:
    args = _parse_args()

    if args.command is None:
        print("Usage: python -m extraction [extract] ...")
        sys.exit(1)

    cfg = _load_cfg(getattr(args, "config", None))

    if args.command == "extract":
        _run_extract(args.pdf, cfg, getattr(args, "output", None))
        return

    print(f"Unknown command: {args.command}")
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest extraction/tests/test_cli.py -v`
Expected: both tests PASS

- [ ] **Step 5: Run existing test suite to verify no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: all existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add extraction/__main__.py extraction/tests/test_cli.py
git commit -m "feat(extraction): add CLI entry point for extraction block"
```

---

## Post-Plan: What Comes Next

This plan builds the **framework**. After completion, the following work remains (separate plans):

1. **Port ML adapters** — MinerU segmenter, OlmOCR text extractor, PPFormulaNet formula extractor, Qwen2.5-VL figure descriptor. Each adapter is a new file in `extraction/adapters/`.
2. **Section path assignment** — build the `section_path` for each element from segmentation data (heading detection + hierarchy).
3. **Relation extraction** — populate `document_rich.json` relations (text-to-table references, etc.).
4. **Multi-page element merging** — detect and merge tables that span pages.
5. **Push old code + restructure** — push current `src/` to GitHub, then clean up or archive.
