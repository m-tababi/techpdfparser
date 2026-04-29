"""Render an Azure Document Intelligence (prebuilt-layout) JSON as a single HTML file.

Usage:
    python scripts/render_ms_layout_html.py <layout.json> [<output.html>]
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any


ROLE_TAG = {
    "title": "h1",
    "sectionHeading": "h3",
    "pageHeader": "div",
    "pageFooter": "div",
    "pageNumber": "div",
    "footnote": "div",
}
ROLE_CLASS = {
    "title": "title",
    "sectionHeading": "section-heading",
    "pageHeader": "page-meta header",
    "pageFooter": "page-meta footer",
    "pageNumber": "page-meta number",
    "footnote": "footnote",
}


def parse_ref(ref: str) -> tuple[str, int] | None:
    if not ref.startswith("/"):
        return None
    parts = ref.strip("/").split("/")
    if len(parts) != 2:
        return None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return None


def collect_para_refs(elements: list[str]) -> list[int]:
    out: list[int] = []
    for e in elements or []:
        ref = parse_ref(e)
        if ref and ref[0] == "paragraphs":
            out.append(ref[1])
    return out


def page_of(para: dict[str, Any]) -> int:
    regions = para.get("boundingRegions") or []
    if regions:
        return int(regions[0]["pageNumber"])
    return 0


def render_table(t: dict[str, Any], idx: int) -> str:
    rows = t["rowCount"]
    cols = t["columnCount"]
    grid: list[list[str | None]] = [[None] * cols for _ in range(rows)]
    spans: list[list[tuple[int, int]]] = [[(1, 1)] * cols for _ in range(rows)]
    skip = [[False] * cols for _ in range(rows)]
    header_row = set()
    for cell in t["cells"]:
        r, c = cell["rowIndex"], cell["columnIndex"]
        rs = cell.get("rowSpan", 1)
        cs = cell.get("columnSpan", 1)
        grid[r][c] = cell.get("content", "")
        spans[r][c] = (rs, cs)
        if cell.get("kind") == "columnHeader":
            header_row.add(r)
        for dr in range(rs):
            for dc in range(cs):
                if dr == 0 and dc == 0:
                    continue
                if r + dr < rows and c + dc < cols:
                    skip[r + dr][c + dc] = True

    page = t["boundingRegions"][0]["pageNumber"] if t.get("boundingRegions") else "?"
    parts = [
        f'<figure class="ms-table" id="table-{idx}">',
        f'<figcaption>Tabelle {idx + 1} &middot; Seite {page} &middot; {rows}&times;{cols}</figcaption>',
        "<table>",
    ]
    for r in range(rows):
        parts.append("<tr>")
        for c in range(cols):
            if skip[r][c]:
                continue
            content = html.escape((grid[r][c] or "").strip())
            rs, cs = spans[r][c]
            attrs = []
            if rs > 1:
                attrs.append(f'rowspan="{rs}"')
            if cs > 1:
                attrs.append(f'colspan="{cs}"')
            tag = "th" if r in header_row else "td"
            parts.append(f"<{tag}{(' ' + ' '.join(attrs)) if attrs else ''}>{content}</{tag}>")
        parts.append("</tr>")
    parts.append("</table></figure>")
    return "\n".join(parts)


def render_figure(f: dict[str, Any], idx: int, paragraphs: list[dict[str, Any]]) -> str:
    page = f["boundingRegions"][0]["pageNumber"] if f.get("boundingRegions") else "?"
    poly = f["boundingRegions"][0].get("polygon", []) if f.get("boundingRegions") else []
    bbox = ""
    if len(poly) == 8:
        xs = poly[0::2]
        ys = poly[1::2]
        bbox = f"x:{min(xs):.2f}–{max(xs):.2f} y:{min(ys):.2f}–{max(ys):.2f} (inch)"
    refs = collect_para_refs(f.get("elements", []))
    text_lines = []
    for r in refs:
        if 0 <= r < len(paragraphs):
            content = paragraphs[r].get("content", "").strip()
            if content:
                text_lines.append(html.escape(content))
    inner = "<br>".join(text_lines) if text_lines else "<em>(keine Text-Elemente)</em>"
    fid = f.get("id", str(idx))
    return (
        f'<figure class="ms-figure" id="figure-{idx}">'
        f'<figcaption>Figure {fid} &middot; Seite {page} &middot; {bbox}</figcaption>'
        f'<div class="figure-text">{inner}</div>'
        "</figure>"
    )


def render_paragraph(p: dict[str, Any]) -> str:
    role = p.get("role")
    text = html.escape(p.get("content", ""))
    if role and role in ROLE_TAG:
        tag = ROLE_TAG[role]
        cls = ROLE_CLASS[role]
        badge = f'<span class="role-badge">{role}</span>' if role not in ("title", "sectionHeading") else ""
        return f'<{tag} class="{cls}">{badge}{text}</{tag}>'
    return f"<p>{text}</p>"


def build_html(data: dict[str, Any], src_name: str) -> str:
    ar = data["analyzeResult"]
    paragraphs: list[dict[str, Any]] = ar.get("paragraphs", [])
    tables: list[dict[str, Any]] = ar.get("tables", [])
    figures: list[dict[str, Any]] = ar.get("figures", [])
    pages: list[dict[str, Any]] = ar.get("pages", [])

    # Map paragraph-idx -> ("table"|"figure", obj_idx); pick first occurrence as anchor
    para_to_block: dict[int, tuple[str, int]] = {}
    block_first_para: dict[tuple[str, int], int] = {}

    def assign(kind: str, obj_idx: int, refs: list[int]) -> None:
        if not refs:
            return
        first = min(refs)
        block_first_para[(kind, obj_idx)] = first
        for r in refs:
            para_to_block.setdefault(r, (kind, obj_idx))

    for ti, t in enumerate(tables):
        refs: list[int] = []
        for c in t.get("cells", []):
            refs.extend(collect_para_refs(c.get("elements", [])))
        assign("table", ti, refs)

    for fi, f in enumerate(figures):
        refs = collect_para_refs(f.get("elements", []))
        assign("figure", fi, refs)

    # Build page-grouped output
    out_pages: list[tuple[int, list[str]]] = []
    current_page: int | None = None
    section_anchors: list[tuple[str, int, str]] = []  # (text, page, anchor_id)

    rendered_blocks: set[tuple[str, int]] = set()

    def flush_block_at(idx: int, parts: list[str]) -> bool:
        """If paragraph idx anchors a block we haven't rendered, render it and return True."""
        kind_obj = para_to_block.get(idx)
        if not kind_obj:
            return False
        if kind_obj in rendered_blocks:
            return True
        first = block_first_para.get(kind_obj)
        if first != idx:
            # Belongs to a block but isn't its anchor — suppress (already / will be rendered there).
            return True
        rendered_blocks.add(kind_obj)
        kind, obj_idx = kind_obj
        if kind == "table":
            parts.append(render_table(tables[obj_idx], obj_idx))
        else:
            parts.append(render_figure(figures[obj_idx], obj_idx, paragraphs))
        return True

    parts: list[str] = []
    for pi, para in enumerate(paragraphs):
        page = page_of(para)
        if current_page is None or page != current_page:
            if current_page is not None:
                out_pages.append((current_page, parts))
                parts = []
            current_page = page

        if pi in para_to_block:
            # Belongs to a block — render block once at anchor, otherwise skip.
            flush_block_at(pi, parts)
            continue

        if para.get("role") == "sectionHeading":
            anchor = f"sec-{len(section_anchors)}"
            section_anchors.append((para.get("content", ""), page, anchor))
            text = html.escape(para.get("content", ""))
            parts.append(f'<h3 class="section-heading" id="{anchor}">{text}</h3>')
            continue

        parts.append(render_paragraph(para))

    if current_page is not None:
        out_pages.append((current_page, parts))

    # Stats
    role_counts: dict[str, int] = {}
    for p in paragraphs:
        r = p.get("role") or "(text)"
        role_counts[r] = role_counts.get(r, 0) + 1
    stats_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in sorted(role_counts.items(), key=lambda kv: -kv[1])
    )

    toc_items = "".join(
        f'<li><a href="#{a}">{html.escape(t)} <span class="muted">(S. {p})</span></a></li>'
        for t, p, a in section_anchors
    )

    page_blocks = []
    for page, html_parts in out_pages:
        page_blocks.append(
            f'<section class="page" id="page-{page}">'
            f'<header class="page-bar"><span>Seite {page}</span>'
            f'<a href="#top" class="muted">↑ top</a></header>'
            + "\n".join(html_parts)
            + "</section>"
        )

    return f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>MS Layout &mdash; {html.escape(src_name)}</title>
<style>
:root {{
  --fg: #1a1a1a; --muted: #666; --bg: #fafafa; --border: #ddd;
  --accent: #0b6cb8; --table-head: #eef3f8; --figure-bg: #f4f0e8;
  --header-bg: #f5f5f5; --footer-bg: #f5f5f5;
}}
* {{ box-sizing: border-box; }}
body {{
  font: 14px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif;
  margin: 0; color: var(--fg); background: var(--bg);
}}
.layout {{ display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }}
.sidebar {{
  position: sticky; top: 0; align-self: start; height: 100vh;
  overflow-y: auto; background: #fff; border-right: 1px solid var(--border);
  padding: 16px;
}}
.sidebar h1 {{ font-size: 14px; margin: 0 0 8px; }}
.sidebar h2 {{ font-size: 12px; text-transform: uppercase; color: var(--muted); margin: 16px 0 6px; letter-spacing: .05em; }}
.sidebar ul {{ list-style: none; padding: 0; margin: 0; font-size: 13px; }}
.sidebar li {{ margin: 2px 0; }}
.sidebar a {{ color: var(--accent); text-decoration: none; }}
.sidebar a:hover {{ text-decoration: underline; }}
.sidebar table {{ font-size: 12px; width: 100%; border-collapse: collapse; }}
.sidebar td {{ padding: 2px 4px; border-bottom: 1px dotted var(--border); }}
.muted {{ color: var(--muted); font-weight: normal; }}
main {{ padding: 16px 32px 64px; max-width: 880px; }}
.page {{ background: #fff; border: 1px solid var(--border); border-radius: 6px;
  padding: 18px 24px; margin: 16px 0; }}
.page-bar {{ display: flex; justify-content: space-between; align-items: baseline;
  border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 12px;
  font-weight: 600; color: #444; }}
.page-bar a {{ font-size: 12px; text-decoration: none; }}
h1.title {{ font-size: 22px; margin: 8px 0 16px; }}
h3.section-heading {{ font-size: 15px; margin: 18px 0 6px; color: #222;
  border-left: 3px solid var(--accent); padding-left: 8px; }}
p {{ margin: 6px 0; }}
.page-meta {{ font-size: 11px; color: var(--muted); padding: 2px 6px; margin: 4px 0;
  border-radius: 3px; display: inline-block; }}
.page-meta.header {{ background: var(--header-bg); }}
.page-meta.footer {{ background: var(--footer-bg); }}
.page-meta.number {{ background: #eef; }}
.role-badge {{ font-size: 9px; text-transform: uppercase; background: #ddd;
  color: #333; padding: 1px 4px; border-radius: 2px; margin-right: 4px;
  vertical-align: middle; letter-spacing: .04em; }}
.footnote {{ font-size: 12px; color: #555; border-top: 1px dashed var(--border);
  padding-top: 4px; margin-top: 8px; }}
.ms-table {{ margin: 12px 0; padding: 0; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }}
.ms-table figcaption {{ background: var(--table-head); padding: 4px 8px; font-size: 12px;
  font-weight: 600; color: #345; }}
.ms-table table {{ border-collapse: collapse; width: 100%; }}
.ms-table td, .ms-table th {{ border: 1px solid var(--border); padding: 4px 8px;
  font-size: 13px; vertical-align: top; }}
.ms-table th {{ background: var(--table-head); text-align: left; }}
.ms-figure {{ margin: 12px 0; padding: 8px 12px; background: var(--figure-bg);
  border: 1px dashed #c7b89c; border-radius: 4px; }}
.ms-figure figcaption {{ font-size: 12px; font-weight: 600; color: #6a5a3a; margin-bottom: 4px; }}
.figure-text {{ font-size: 13px; color: #444; }}
</style>
</head>
<body>
<a id="top"></a>
<div class="layout">
<aside class="sidebar">
<h1>MS Layout JSON</h1>
<div class="muted" style="font-size:12px">{html.escape(src_name)}</div>
<div class="muted" style="font-size:12px;margin-top:4px">
  Modell: {html.escape(ar.get('modelId',''))} &middot; API {html.escape(ar.get('apiVersion',''))}<br>
  {len(pages)} Seiten &middot; {len(paragraphs)} Paragraphs &middot;
  {len(tables)} Tabellen &middot; {len(figures)} Figures
</div>
<h2>Seiten</h2>
<ul>{''.join(f'<li><a href="#page-{p}">Seite {p}</a></li>' for p, _ in out_pages)}</ul>
<h2>Inhalt (Headings)</h2>
<ul>{toc_items}</ul>
<h2>Paragraph-Rollen</h2>
<table>{stats_rows}</table>
</aside>
<main>
{''.join(page_blocks)}
</main>
</div>
</body>
</html>
"""


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    src = Path(argv[1])
    dst = Path(argv[2]) if len(argv) > 2 else src.with_suffix(".html")
    data = json.loads(src.read_text(encoding="utf-8"))
    html_text = build_html(data, src.name)
    dst.write_text(html_text, encoding="utf-8")
    print(f"wrote {dst} ({dst.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
