"""MinerU 2.5 rich segmenter.

Wraps ``mineru.cli.common.do_parse`` and converts the produced
``*_middle.json`` into the new ``Region`` schema. Unlike the CPU
baseline, this segmenter already carries element content (table
markdown, formula LaTeX, heading text) in each Region — the pipeline's
merge rule will then keep that content instead of calling the dedicated
extractor over the crop.

Requires ``pip install mineru`` plus the supporting model weights.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..models import ElementContent, ElementType, Region
from ..registry import register_segmenter

_BLOCK_TEXT = "text"
_BLOCK_TITLE = "title"
_BLOCK_TABLE = "table"
_BLOCK_CHART = "chart"
_BLOCK_IMAGE = "image"
_BLOCK_INTERLINE_EQUATION = "interline_equation"

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
_SPAN_INTERLINE_EQUATION = "interline_equation"


@register_segmenter("mineru25")
class MinerU25Segmenter:
    TOOL_NAME = "mineru25"

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._do_parse = None

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

        output_dir = Path(tempfile.mkdtemp(prefix="techpdf_mineru_"))
        pdf_bytes = pdf_path.read_bytes()
        self._do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[pdf_path.name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[""],
            backend="pipeline",
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
        for page_number, block in _iter_para_blocks(raw):
            region = _block_to_region(block, page_number)
            if region is not None:
                regions.append(region)
        return regions


def _block_to_region(block: dict[str, Any], page_number: int) -> Region | None:
    block_type = block.get("type")
    bbox = _to_bbox(block.get("bbox"))
    if bbox is None:
        return None

    if block_type == _BLOCK_TEXT:
        text = _lines_to_text(block.get("lines", []))
        if not text:
            return None
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=ElementType.TEXT,
            confidence=1.0,
            content=ElementContent(text=text),
        )

    if block_type == _BLOCK_TITLE:
        text = _lines_to_text(block.get("lines", []))
        if not text:
            return None
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=ElementType.HEADING,
            confidence=1.0,
            content=ElementContent(text=text),
        )

    if block_type == _BLOCK_TABLE:
        _, html = _extract_body_data(block)
        rows = _html_to_rows(html)
        caption = _collect_block_text(block, _BLOCK_TABLE_CAPTION)
        markdown = _rows_to_markdown(rows) if rows else html.strip()
        if not markdown and not caption:
            return None
        content = ElementContent(
            markdown=markdown or None,
            text=markdown or None,
            caption=caption or None,
        )
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=ElementType.TABLE,
            confidence=1.0,
            content=content,
        )

    if block_type == _BLOCK_INTERLINE_EQUATION:
        _, latex = _extract_body_data(block)
        latex = latex.strip()
        if not latex:
            return None
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=ElementType.FORMULA,
            confidence=1.0,
            content=ElementContent(latex=latex, text=latex),
        )

    if block_type in {_BLOCK_IMAGE, _BLOCK_CHART}:
        caption_type = (
            _BLOCK_CHART_CAPTION if block_type == _BLOCK_CHART else _BLOCK_IMAGE_CAPTION
        )
        caption = _collect_block_text(block, caption_type) or None
        # image_path is set by the pipeline after cropping the page image;
        # MinerU's internal PNG under parse_dir/images is not carried over.
        region_kind = (
            ElementType.DIAGRAM if block_type == _BLOCK_CHART else ElementType.FIGURE
        )
        return Region(
            page=page_number,
            bbox=bbox,
            region_type=region_kind,
            confidence=1.0,
            content=ElementContent(caption=caption) if caption else None,
        )

    return None


def _find_middle_json(output_dir: Path) -> Path:
    for path in output_dir.rglob("*_middle.json"):
        return path
    raise FileNotFoundError("MinerU did not produce a *_middle.json artifact")


def _iter_para_blocks(
    raw: dict[str, Any],
) -> Iterator[tuple[int, dict[str, Any]]]:
    for page_idx, page in enumerate(raw.get("pdf_info", [])):
        page_number = int(page.get("page_idx", page_idx))
        for block in page.get("para_blocks") or []:
            yield page_number, block


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
                if span_type == _SPAN_INTERLINE_EQUATION:
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


def _lines_to_text(lines: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for line in lines:
        spans: list[str] = []
        for span in line.get("spans", []):
            span_type = span.get("type")
            if span_type not in {_SPAN_TEXT, _SPAN_INLINE_EQUATION}:
                continue
            content = str(span.get("content", "")).strip()
            if content:
                spans.append(content)
        if spans:
            chunks.append(" ".join(spans))
    return "\n".join(chunks).strip()


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
