from __future__ import annotations

import json
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ...core.models.document import BoundingBox
from ...core.models.elements import Figure, Formula, Table
from ...core.registry import register_structured_parser
from ...utils.ids import generate_element_id

_BLOCK_IMAGE = "image"
_BLOCK_TABLE = "table"
_BLOCK_CHART = "chart"
_BLOCK_IMAGE_BODY = "image_body"
_BLOCK_TABLE_BODY = "table_body"
_BLOCK_CHART_BODY = "chart_body"
_BLOCK_IMAGE_CAPTION = "image_caption"
_BLOCK_TABLE_CAPTION = "table_caption"
_BLOCK_CHART_CAPTION = "chart_caption"
_BLOCK_IMAGE_FOOTNOTE = "image_footnote"
_BLOCK_TABLE_FOOTNOTE = "table_footnote"
_BLOCK_CHART_FOOTNOTE = "chart_footnote"
_BLOCK_INTERLINE_EQUATION = "interline_equation"

_SPAN_TEXT = "text"
_SPAN_INLINE_EQUATION = "inline_equation"
_SPAN_IMAGE = "image"
_SPAN_TABLE = "table"
_SPAN_CHART = "chart"
_SPAN_INTERLINE_EQUATION = "interline_equation"


@register_structured_parser("mineru25")
class MinerU25Parser:
    """Structured parser using the current MinerU pipeline entrypoint.

    Recent MinerU releases no longer expose ``mineru.pipeline.PDFPipeline``.
    Instead, the supported Python entrypoint is ``mineru.cli.common.do_parse``,
    which writes a ``*_middle.json`` artifact that we convert into our internal
    Table / Formula / Figure models.

    Model: opendatalab/MinerU
    Requires: pip install mineru
    """

    TOOL_NAME = "mineru25"
    TOOL_VERSION = "2.5"

    def __init__(self, model_path: str = "", device: str = "cuda") -> None:
        self._model_path = model_path
        self._device = device
        self._do_parse = None

    def _load(self) -> None:
        if self._do_parse is not None:
            return
        try:
            from mineru.cli.common import do_parse
        except ImportError as exc:
            raise ImportError("mineru not installed. Run: pip install mineru") from exc
        self._do_parse = do_parse

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    @property
    def tool_version(self) -> str:
        try:
            return version("mineru")
        except PackageNotFoundError:
            return self.TOOL_VERSION

    def parse(
        self, pdf_path: Path, doc_id: str
    ) -> tuple[list[Table], list[Formula], list[Figure]]:
        self._load()

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
        raw = json.loads(middle_json_path.read_text())
        parse_dir = middle_json_path.parent
        source = str(pdf_path)

        tables: list[Table] = []
        formulas: list[Formula] = []
        figures: list[Figure] = []
        table_seq = formula_seq = figure_seq = 0

        for page_number, block in _iter_structured_blocks(raw):
            block_type = block.get("type")
            if block_type == _BLOCK_TABLE:
                table = self._table_from_block(
                    block, parse_dir, doc_id, page_number, source, table_seq
                )
                if table is not None:
                    tables.append(table)
                    table_seq += 1
            elif block_type == _BLOCK_INTERLINE_EQUATION:
                formula = self._formula_from_block(
                    block, parse_dir, doc_id, page_number, source, formula_seq
                )
                if formula is not None:
                    formulas.append(formula)
                    formula_seq += 1
            elif block_type in {_BLOCK_IMAGE, _BLOCK_CHART}:
                figure = self._figure_from_block(
                    block, parse_dir, doc_id, page_number, source, figure_seq
                )
                if figure is not None:
                    figures.append(figure)
                    figure_seq += 1

        return tables, formulas, figures

    def _table_from_block(
        self,
        block: dict[str, Any],
        parse_dir: Path,
        doc_id: str,
        page_number: int,
        source: str,
        seq: int,
    ) -> Table | None:
        image_name, html = _extract_body_data(block)
        rows = _html_to_rows(html)
        headers = rows[0] if rows else []
        caption = _collect_block_text(block, _BLOCK_TABLE_CAPTION)
        footnote = _collect_block_text(block, _BLOCK_TABLE_FOOTNOTE)
        content = _table_content(html=html, rows=rows, caption=caption, footnote=footnote)
        if not content and not image_name:
            return None

        return Table(
            object_id=generate_element_id(doc_id, page_number, "table", self.TOOL_NAME, seq),
            doc_id=doc_id,
            source_file=source,
            page_number=page_number,
            tool_name=self.TOOL_NAME,
            tool_version=self.tool_version,
            bbox=_to_bbox(block.get("bbox")),
            content=content,
            rows=rows,
            headers=headers,
        )

    def _formula_from_block(
        self,
        block: dict[str, Any],
        parse_dir: Path,
        doc_id: str,
        page_number: int,
        source: str,
        seq: int,
    ) -> Formula | None:
        image_name, latex = _extract_body_data(block)
        image_path = _resolve_asset_path(parse_dir, image_name)
        text = latex.strip()
        if not text and image_path is None:
            return None

        return Formula(
            object_id=generate_element_id(
                doc_id, page_number, "formula", self.TOOL_NAME, seq
            ),
            doc_id=doc_id,
            source_file=source,
            page_number=page_number,
            tool_name=self.TOOL_NAME,
            tool_version=self.tool_version,
            bbox=_to_bbox(block.get("bbox")),
            latex=text,
            content=text,
            image_path=str(image_path) if image_path else None,
            raw_output_path=str(image_path) if image_path else None,
        )

    def _figure_from_block(
        self,
        block: dict[str, Any],
        parse_dir: Path,
        doc_id: str,
        page_number: int,
        source: str,
        seq: int,
    ) -> Figure | None:
        image_name, _ = _extract_body_data(block)
        image_path = _resolve_asset_path(parse_dir, image_name)
        if image_path is None:
            return None

        caption_type = (
            _BLOCK_CHART_CAPTION if block.get("type") == _BLOCK_CHART else _BLOCK_IMAGE_CAPTION
        )
        footnote_type = (
            _BLOCK_CHART_FOOTNOTE
            if block.get("type") == _BLOCK_CHART
            else _BLOCK_IMAGE_FOOTNOTE
        )
        caption = _combine_texts(
            _collect_block_text(block, caption_type),
            _collect_block_text(block, footnote_type),
        )

        return Figure(
            object_id=generate_element_id(doc_id, page_number, "figure", self.TOOL_NAME, seq),
            doc_id=doc_id,
            source_file=source,
            page_number=page_number,
            tool_name=self.TOOL_NAME,
            tool_version=self.tool_version,
            bbox=_to_bbox(block.get("bbox")),
            image_path=str(image_path),
            caption=caption or None,
            raw_output_path=str(image_path),
        )


def _find_middle_json(output_dir: Path) -> Path:
    for path in output_dir.rglob("*_middle.json"):
        return path
    raise FileNotFoundError("MinerU did not produce a *_middle.json artifact")


def _iter_structured_blocks(raw: dict[str, Any]):
    for page_idx, page in enumerate(raw.get("pdf_info", [])):
        page_number = int(page.get("page_idx", page_idx))
        blocks = [*(page.get("para_blocks") or []), *(page.get("discarded_blocks") or [])]
        for block in blocks:
            yield page_number, block


def _extract_body_data(para_block: dict[str, Any]) -> tuple[str, str]:
    def get_data_from_spans(lines: list[dict[str, Any]]) -> tuple[str, str]:
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
            if block.get("type") in {
                _BLOCK_IMAGE_BODY,
                _BLOCK_TABLE_BODY,
                _BLOCK_CHART_BODY,
            }:
                image_name, content = get_data_from_spans(block.get("lines", []))
                if image_name or content or block.get("type") == _BLOCK_CHART_BODY:
                    return image_name, content
        return "", ""

    return get_data_from_spans(para_block.get("lines", []))


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


def _resolve_asset_path(parse_dir: Path, image_name: str) -> Path | None:
    if not image_name:
        return None
    candidate = Path(image_name)
    if candidate.is_absolute():
        return candidate
    return parse_dir / "images" / candidate.name


def _table_content(
    html: str,
    rows: list[list[str]],
    caption: str,
    footnote: str,
) -> str:
    body = _rows_to_markdown(rows) if rows else html.strip()
    return _combine_texts(caption, body, footnote)


def _combine_texts(*parts: str) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(cleaned)


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


def _to_bbox(raw: list[Any] | None) -> BoundingBox | None:
    if not raw or len(raw) < 4:
        return None
    return BoundingBox(x0=raw[0], y0=raw[1], x1=raw[2], y1=raw[3])
