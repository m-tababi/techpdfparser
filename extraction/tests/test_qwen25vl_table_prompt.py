"""Unit-Tests für die HTML-Extraktion und den Prompt des Qwen-Tabellen-Extractors.

Lädt das Modell nicht — testet nur die Text-Verarbeitung.
"""
from __future__ import annotations

from extraction.adapters.qwen25vl_table import (
    _TABLE_PROMPT,
    _extract_table_html,
)


def test_prompt_demands_html_table_only():
    p = _TABLE_PROMPT.lower()
    assert "html" in p and "<table>" in p
    assert "<th>" in p and "<td>" in p
    # Markdown ausdrücklich verboten, damit der Output deterministisch HTML ist.
    assert "markdown" in p
    assert "rowspan" in p and "colspan" in p


def test_extract_strips_html_code_fence():
    raw = "```html\n<table><tr><th>a</th></tr></table>\n```"
    assert _extract_table_html(raw) == "<table><tr><th>a</th></tr></table>"


def test_extract_strips_bare_code_fence():
    raw = "```\n<table><tr><td>x</td></tr></table>\n```"
    assert _extract_table_html(raw) == "<table><tr><td>x</td></tr></table>"


def test_extract_picks_first_table_when_prose_around():
    raw = "Sure! Here you go:\n<table><tr><td>x</td></tr></table>\nHope that helps."
    assert _extract_table_html(raw) == "<table><tr><td>x</td></tr></table>"


def test_extract_wraps_bare_rows_in_table():
    raw = "<tr><td>1</td></tr><tr><td>2</td></tr>"
    out = _extract_table_html(raw)
    assert out.startswith("<table>") and out.endswith("</table>")
    assert "<tr><td>1</td></tr>" in out


def test_extract_empty_input_is_empty():
    assert _extract_table_html("") == ""
