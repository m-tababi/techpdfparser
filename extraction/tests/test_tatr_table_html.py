"""Unit-Tests für die HTML-Synthese aus TATR-Predictions.

Lädt das Modell nicht — synthetisiert Predictions + Tokens und prüft den
gebauten HTML-String.
"""
from __future__ import annotations

from extraction.adapters.tatr_table import _predictions_to_html


def _row(y0: float, y1: float) -> dict:
    return {"label": "table row", "score": 0.9, "bbox": [0.0, y0, 100.0, y1]}


def _col(x0: float, x1: float) -> dict:
    return {"label": "table column", "score": 0.9, "bbox": [x0, 0.0, x1, 100.0]}


def _col_header(y0: float, y1: float) -> dict:
    return {
        "label": "table column header",
        "score": 0.9,
        "bbox": [0.0, y0, 100.0, y1],
    }


def _spanning(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "label": "table spanning cell",
        "score": 0.9,
        "bbox": [x0, y0, x1, y1],
    }


def _tok(text: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"text": text, "bbox": [x0, y0, x1, y1]}


def test_no_rows_or_columns_returns_empty_string():
    assert _predictions_to_html([], []) == ""


def test_basic_two_by_two_with_column_header():
    preds = [
        _row(0, 20),
        _row(20, 40),
        _col(0, 50),
        _col(50, 100),
        _col_header(0, 20),
    ]
    tokens = [
        _tok("A", 5, 5, 15, 15),
        _tok("B", 55, 5, 65, 15),
        _tok("1", 5, 25, 15, 35),
        _tok("2", 55, 25, 65, 35),
    ]
    html = _predictions_to_html(preds, tokens)
    assert html.startswith("<table>") and html.endswith("</table>")
    # Erste Zeile = Header → <th>, zweite Zeile = Daten → <td>.
    assert "<tr><th>A</th><th>B</th></tr>" in html
    assert "<tr><td>1</td><td>2</td></tr>" in html


def test_spanning_cell_emits_colspan_once():
    preds = [
        _row(0, 20),
        _row(20, 40),
        _col(0, 50),
        _col(50, 100),
        _spanning(0, 0, 100, 20),  # Top row spans both columns
    ]
    tokens = [_tok("Title", 30, 5, 70, 15), _tok("x", 5, 25, 15, 35)]
    html = _predictions_to_html(preds, tokens)
    # Das Spanning ergibt EINE Zelle mit colspan=2 in Zeile 0, nicht zwei.
    assert 'colspan="2"' in html
    assert html.count("<tr>") == 2
    # Zeile 0 hat genau eine Zelle, Zeile 1 hat zwei.
    rows = html.split("<tr>")[1:]
    assert rows[0].count("</td>") + rows[0].count("</th>") == 1
    assert rows[1].count("</td>") + rows[1].count("</th>") == 2


def test_html_escapes_special_chars():
    preds = [_row(0, 20), _col(0, 100)]
    tokens = [_tok("a<b&c>d", 5, 5, 95, 15)]
    html = _predictions_to_html(preds, tokens)
    assert "a&lt;b&amp;c&gt;d" in html
    assert "<b" not in html.split("<tr>", 1)[1].split("</tr>")[0].replace(
        "<th>", ""
    ).replace("<td>", "")
