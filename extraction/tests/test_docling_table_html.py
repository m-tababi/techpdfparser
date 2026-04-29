"""Unit-Tests für die HTML-Synthese aus TableFormer-Cell-Dicts.

Lädt das Modell nicht — synthetisiert Cell-Listen und prüft den gebauten
HTML-String.
"""
from __future__ import annotations

from extraction.adapters.docling_table import _cells_to_html


def _cell(
    *,
    row: int,
    col: int,
    text: str,
    column_header: bool = False,
    row_header: bool = False,
    row_span: int = 1,
    col_span: int = 1,
) -> dict:
    return {
        "start_row_offset_idx": row,
        "start_col_offset_idx": col,
        "row_span": row_span,
        "col_span": col_span,
        "column_header": column_header,
        "row_header": row_header,
        "text": text,
        "bbox": {"l": 0, "t": 0, "r": 0, "b": 0},
    }


def test_empty_cells_returns_empty_string():
    assert _cells_to_html([], []) == ""


def test_basic_two_by_two_with_column_header_row():
    cells = [
        _cell(row=0, col=0, text="A", column_header=True),
        _cell(row=0, col=1, text="B", column_header=True),
        _cell(row=1, col=0, text="1"),
        _cell(row=1, col=1, text="2"),
    ]
    html = _cells_to_html(cells, [])
    assert html.startswith("<table>") and html.endswith("</table>")
    assert "<tr><th>A</th><th>B</th></tr>" in html
    assert "<tr><td>1</td><td>2</td></tr>" in html


def test_row_header_is_th():
    cells = [
        _cell(row=0, col=0, text="ID", row_header=True),
        _cell(row=0, col=1, text="42"),
    ]
    html = _cells_to_html(cells, [])
    assert "<tr><th>ID</th><td>42</td></tr>" in html


def test_row_span_and_col_span_emitted():
    cells = [
        _cell(row=0, col=0, text="merged", col_span=2, row_span=2),
        _cell(row=2, col=0, text="x"),
    ]
    html = _cells_to_html(cells, [])
    assert 'rowspan="2"' in html and 'colspan="2"' in html
    assert "<td>x</td>" in html


def test_html_escapes_special_chars():
    cells = [_cell(row=0, col=0, text="a<b&c>d")]
    html = _cells_to_html(cells, [])
    assert "a&lt;b&amp;c&gt;d" in html


def _cell_no_offsets(
    *,
    bbox: dict,
    text: str,
    column_header: bool = False,
    row_header: bool = False,
    row_span: int = 1,
    col_span: int = 1,
) -> dict:
    """A cell dict mimicking what high-level multi_table_predict returns:
    no start_row/col_offset_idx fields; row/column must be derived from bbox.
    """
    return {
        "row_span": row_span,
        "col_span": col_span,
        "column_header": column_header,
        "row_header": row_header,
        "text": text,
        "bbox": bbox,
    }


def test_falls_back_to_bbox_when_offsets_missing():
    # Two-row, two-column table where cells expose only bbox + flags.
    cells = [
        _cell_no_offsets(
            bbox={"l": 0, "t": 0, "r": 50, "b": 20}, text="A", column_header=True
        ),
        _cell_no_offsets(
            bbox={"l": 50, "t": 0, "r": 100, "b": 20}, text="B", column_header=True
        ),
        _cell_no_offsets(bbox={"l": 0, "t": 20, "r": 50, "b": 40}, text="1"),
        _cell_no_offsets(bbox={"l": 50, "t": 20, "r": 100, "b": 40}, text="2"),
    ]
    html = _cells_to_html(cells, [])
    # Without the fallback, all cells would land in row 0 and we'd get a
    # single-row HTML table. Verify two distinct rows in the output.
    assert html.count("<tr>") == 2
    assert "<tr><th>A</th><th>B</th></tr>" in html
    assert "<tr><td>1</td><td>2</td></tr>" in html
