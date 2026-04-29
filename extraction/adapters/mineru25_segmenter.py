"""MinerU rich segmenters.

Wraps ``mineru.cli.common.do_parse`` and converts the produced
``*_middle.json`` into the new ``Region`` schema. Unlike the CPU
baseline, this segmenter already carries element content (table
markdown, formula LaTeX, heading text) in each Region — the pipeline's
merge rule will then keep that content instead of calling the dedicated
extractor over the crop.

Registered adapter names:

* ``mineru25`` uses MinerU's stable ``pipeline`` backend.
* ``mineru_hybrid`` uses ``hybrid-auto-engine``.
* ``mineru_vlm`` uses ``vlm-auto-engine``.

Requires ``pip install mineru`` plus the supporting model weights. The project
currently depends on ``mineru>=3.1``.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from ..models import ElementContent, ElementType, Region
from ..registry import (
    register_formula_extractor,
    register_segmenter,
    register_table_extractor,
    register_text_extractor,
)

_BLOCK_TEXT = "text"
_BLOCK_TITLE = "title"
_BLOCK_TABLE = "table"
_BLOCK_CHART = "chart"
_BLOCK_IMAGE = "image"
_BLOCK_EQUATION = "equation"
_BLOCK_INTERLINE_EQUATION = "interline_equation"
_BLOCK_LIST = "list"
_BLOCK_CODE = "code"
_BLOCK_ALGORITHM = "algorithm"

_BLOCK_IMAGE_BODY = "image_body"
_BLOCK_TABLE_BODY = "table_body"
_BLOCK_CHART_BODY = "chart_body"
_BLOCK_IMAGE_CAPTION = "image_caption"
_BLOCK_TABLE_CAPTION = "table_caption"
_BLOCK_CHART_CAPTION = "chart_caption"

_SPAN_TEXT = "text"
_SPAN_INLINE_EQUATION = "inline_equation"
_SPAN_IMAGE = "image"
_SPAN_TABLE = "table"
_SPAN_CHART = "chart"
_SPAN_EQUATION = "equation"
_SPAN_INTERLINE_EQUATION = "interline_equation"


class _BaseMinerUSegmenter:
    TOOL_NAME = "mineru"
    BACKEND = "pipeline"

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._do_parse: Callable[..., Any] | None = None

    def _load(self) -> None:
        if self._do_parse is not None:
            return
        try:
            from mineru.cli.common import do_parse
        except ImportError as exc:
            raise ImportError(
                "mineru not installed. Run: pip install mineru"
            ) from exc
        self._do_parse = do_parse

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def segment(self, pdf_path: Path) -> list[Region]:
        self._load()
        assert self._do_parse is not None

        pdf_bytes = pdf_path.read_bytes()
        with tempfile.TemporaryDirectory(prefix="techpdf_mineru_") as tmp_dir:
            output_dir = Path(tmp_dir)
            self._do_parse(
                output_dir=str(output_dir),
                pdf_file_names=[pdf_path.name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[""],
                backend=self.BACKEND,
                parse_method="auto",
                formula_enable=True,
                table_enable=True,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=False,
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
            )

            middle_json_path = _find_middle_json(output_dir)
            raw = json.loads(middle_json_path.read_text(encoding="utf-8"))

        regions: list[Region] = []
        for page_number, block, layout_dets in _iter_para_blocks(raw):
            region = _block_to_region(block, page_number, layout_dets)
            if region is not None:
                region.reading_order_index = len(regions)
                regions.append(region)
        return regions


@register_segmenter("mineru25")
class MinerU25Segmenter(_BaseMinerUSegmenter):
    TOOL_NAME = "mineru25"
    BACKEND = "pipeline"


@register_segmenter("mineru_hybrid")
class MinerUHybridSegmenter(_BaseMinerUSegmenter):
    TOOL_NAME = "mineru_hybrid"
    BACKEND = "hybrid-auto-engine"


@register_segmenter("mineru_vlm")
class MinerUVLMSegmenter(_BaseMinerUSegmenter):
    TOOL_NAME = "mineru_vlm"
    BACKEND = "vlm-auto-engine"


class _MinerUPassthroughExtractor:
    TOOL_NAME = "mineru"

    def __init__(self, **_kwargs: Any) -> None:
        # Shares the adapter config block with the segmenter; passthrough
        # extractors have nothing to configure, so swallow any kwargs.
        pass

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def extract(self, _image: Any, _page_number: int) -> ElementContent:
        return ElementContent()


@register_table_extractor("mineru25")
class MinerU25TableExtractor(_MinerUPassthroughExtractor):
    """Passthrough role so the config can name mineru25 as the table extractor.

    The MinerU segmenter already populates table markdown in each Region, so
    the pipeline's merge rule keeps the segmenter content and never calls
    this extractor. It exists to satisfy the registry lookup when the default
    config wires table_extractor = mineru25.
    """

    TOOL_NAME = "mineru25"


@register_text_extractor("mineru25")
class MinerU25TextExtractor(_MinerUPassthroughExtractor):
    """Passthrough role for text/heading regions.

    MinerU's middle_json carries per-region text already, so the pipeline's
    role-match keeps the segmenter content and never calls this extractor.
    """

    TOOL_NAME = "mineru25"


@register_formula_extractor("mineru25")
class MinerU25FormulaExtractor(_MinerUPassthroughExtractor):
    """Passthrough role for interline-equation regions.

    MinerU yields LaTeX for each interline_equation block; the pipeline's
    role-match keeps that content and never calls this extractor.
    """

    TOOL_NAME = "mineru25"


@register_table_extractor("mineru_hybrid")
class MinerUHybridTableExtractor(_MinerUPassthroughExtractor):
    TOOL_NAME = "mineru_hybrid"


@register_text_extractor("mineru_hybrid")
class MinerUHybridTextExtractor(_MinerUPassthroughExtractor):
    TOOL_NAME = "mineru_hybrid"


@register_formula_extractor("mineru_hybrid")
class MinerUHybridFormulaExtractor(_MinerUPassthroughExtractor):
    TOOL_NAME = "mineru_hybrid"


@register_table_extractor("mineru_vlm")
class MinerUVLMTableExtractor(_MinerUPassthroughExtractor):
    TOOL_NAME = "mineru_vlm"


@register_text_extractor("mineru_vlm")
class MinerUVLMTextExtractor(_MinerUPassthroughExtractor):
    TOOL_NAME = "mineru_vlm"


@register_formula_extractor("mineru_vlm")
class MinerUVLMFormulaExtractor(_MinerUPassthroughExtractor):
    TOOL_NAME = "mineru_vlm"


_IOU_MATCH_THRESHOLD = 0.5


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _confidence_for_block(
    block: dict[str, Any], layout_dets: list[dict[str, Any]]
) -> float:
    # MinerU writes the detection score directly onto each para_block.
    # Prefer that. layout_dets is populated only by older backends; when it
    # is, match by IoU and return the best-overlapping detection's score.
    direct_score = block.get("score")
    if direct_score is not None:
        return float(direct_score)

    block_bbox = _to_bbox(block.get("bbox"))
    if block_bbox is None:
        return 1.0
    best_score: float | None = None
    best_iou = 0.0
    for det in layout_dets:
        det_bbox = _to_bbox(det.get("bbox"))
        if det_bbox is None:
            continue
        iou = _bbox_iou(block_bbox, det_bbox)
        if iou < _IOU_MATCH_THRESHOLD or iou <= best_iou:
            continue
        best_iou = iou
        best_score = float(det.get("score", 1.0))
    return 1.0 if best_score is None else best_score


def _block_to_region(
    block: dict[str, Any],
    page_number: int,
    layout_dets: list[dict[str, Any]],
) -> Region | None:
    block_type = block.get("type")
    bbox = _to_bbox(block.get("bbox"))
    if bbox is None:
        return None

    confidence = _confidence_for_block(block, layout_dets)

    if block_type in {_BLOCK_TEXT, _BLOCK_LIST, _BLOCK_CODE, _BLOCK_ALGORITHM}:
        text = _block_to_text(block)
        if not text:
            return None
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=ElementType.TEXT,
            confidence=confidence,
            content=ElementContent(text=text),
        )

    if block_type == _BLOCK_TITLE:
        text = _block_to_text(block)
        if not text:
            return None
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=ElementType.HEADING,
            confidence=confidence,
            content=ElementContent(text=text),
        )

    if block_type == _BLOCK_TABLE:
        _, body = _extract_body_data(block)
        body = body.strip()
        direct_html = _first_text_field(block, ("html",))
        direct_content = _first_text_field(block, ("content", "markdown", "text"))
        html = direct_html
        markdown = ""
        if body:
            if _looks_like_html(body):
                html = html or body
            else:
                markdown = body
        if direct_content:
            if _looks_like_html(direct_content):
                html = html or direct_content
            else:
                markdown = markdown or direct_content
        rows = _html_to_rows(html)
        caption = _caption_text(block, _BLOCK_TABLE_CAPTION, ("caption", "table_caption"))
        # markdown is a lossy flattening (rowspan/colspan ignored); html keeps
        # the hierarchical structure so downstream consumers can choose.
        markdown = markdown or (_rows_to_markdown(rows) if rows else html)
        if not markdown and not caption and not html:
            return None
        content = ElementContent(
            html=html or None,
            markdown=markdown or None,
            text=markdown or None,
            caption=caption or None,
        )
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=ElementType.TABLE,
            confidence=confidence,
            content=content,
        )

    if block_type in {_BLOCK_EQUATION, _BLOCK_INTERLINE_EQUATION}:
        _, latex = _extract_body_data(block)
        latex = (
            latex
            or _first_text_field(block, ("latex", "content", "text"))
            or _block_to_text(block)
        ).strip()
        if not latex:
            return None
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=ElementType.FORMULA,
            confidence=confidence,
            content=ElementContent(latex=latex, text=latex),
        )

    if block_type in {_BLOCK_IMAGE, _BLOCK_CHART}:
        caption_type = (
            _BLOCK_CHART_CAPTION if block_type == _BLOCK_CHART else _BLOCK_IMAGE_CAPTION
        )
        direct_caption_keys = (
            ("caption", "chart_caption")
            if block_type == _BLOCK_CHART
            else ("caption", "image_caption")
        )
        figure_caption = _caption_text(block, caption_type, direct_caption_keys) or None
        # image_path is set by the pipeline after cropping the page image;
        # MinerU's internal PNG under parse_dir/images is not carried over.
        region_kind = (
            ElementType.DIAGRAM if block_type == _BLOCK_CHART else ElementType.FIGURE
        )
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=region_kind,
            confidence=confidence,
            content=ElementContent(caption=figure_caption) if figure_caption else None,
        )

    return None


def _find_middle_json(output_dir: Path) -> Path:
    for path in output_dir.rglob("*_middle.json"):
        return path
    raise FileNotFoundError("MinerU did not produce a *_middle.json artifact")


def _iter_para_blocks(
    raw: dict[str, Any],
) -> Iterator[tuple[int, dict[str, Any], list[dict[str, Any]]]]:
    for page_idx, page in enumerate(raw.get("pdf_info", [])):
        page_number = int(page.get("page_idx", page_idx))
        layout_dets = page.get("layout_dets") or []
        for block in page.get("para_blocks") or page.get("blocks") or []:
            yield page_number, block, layout_dets


def _extract_body_data(para_block: dict[str, Any]) -> tuple[str, str]:
    def from_spans(lines: list[dict[str, Any]]) -> tuple[str, str]:
        for line in lines:
            for span in line.get("spans", []):
                span_type = span.get("type")
                if span_type == _SPAN_TABLE:
                    return str(span.get("image_path", "")), str(span.get("html", ""))
                if span_type == _SPAN_CHART:
                    return str(span.get("image_path", "")), str(span.get("content", ""))
                if span_type == _SPAN_IMAGE:
                    return str(span.get("image_path", "")), ""
                if span_type in {_SPAN_EQUATION, _SPAN_INTERLINE_EQUATION}:
                    return str(span.get("image_path", "")), str(span.get("content", ""))
        return "", ""

    if "blocks" in para_block:
        for block in para_block.get("blocks", []):
            if block.get("type") in {_BLOCK_IMAGE_BODY, _BLOCK_TABLE_BODY, _BLOCK_CHART_BODY}:
                image_name, content = from_spans(block.get("lines", []))
                if image_name or content or block.get("type") == _BLOCK_CHART_BODY:
                    return image_name, content
        return "", ""
    return from_spans(para_block.get("lines", []))


def _collect_block_text(para_block: dict[str, Any], block_type: str) -> str:
    parts: list[str] = []
    for block in para_block.get("blocks", []):
        if block.get("type") != block_type:
            continue
        text = _lines_to_text(block.get("lines", []))
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _caption_text(
    para_block: dict[str, Any],
    nested_type: str,
    direct_keys: tuple[str, ...],
) -> str:
    return _collect_block_text(para_block, nested_type) or _first_text_field(
        para_block, direct_keys
    )


def _block_to_text(block: dict[str, Any]) -> str:
    direct = _first_text_field(block, ("content", "text", "code_body"))
    if direct:
        return direct
    line_text = _lines_to_text(block.get("lines", []))
    if line_text:
        return line_text
    parts = [_block_to_text(child) for child in block.get("blocks", [])]
    return "\n".join(part for part in parts if part).strip()


def _first_text_field(block: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _value_to_text(block.get(key))
        if value:
            return value
    return ""


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value).strip()
    if isinstance(value, list):
        parts = [_value_to_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        return _first_text_field(value, ("content", "text", "code_body"))
    return ""


def _lines_to_text(lines: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for line in lines:
        spans: list[str] = []
        for span in line.get("spans", []):
            span_type = span.get("type")
            if (
                span_type is not None
                and span_type not in {
                    _SPAN_TEXT,
                    _SPAN_INLINE_EQUATION,
                    _SPAN_EQUATION,
                    _SPAN_INTERLINE_EQUATION,
                }
            ):
                continue
            content = str(span.get("content", "")).strip()
            if content:
                spans.append(content)
        if spans:
            chunks.append(" ".join(spans))
    return "\n".join(chunks).strip()


def _looks_like_html(text: str) -> bool:
    return text.lstrip().startswith("<")


def _html_to_rows(html: str) -> list[list[str]]:
    if not html.strip():
        return []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html, "html.parser")
    rows: list[list[str]] = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            cells = tr.find_all(["th", "td"])
        row = [cell.get_text(" ", strip=True) for cell in cells]
        if row:
            rows.append(row)
    return rows


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines = [" | ".join(rows[0])]
    lines.append(" | ".join("---" for _ in rows[0]))
    lines.extend(" | ".join(row) for row in rows[1:])
    return "\n".join(lines)


def _to_bbox(raw: list[Any] | None) -> list[float] | None:
    if not raw or len(raw) < 4:
        return None
    return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
