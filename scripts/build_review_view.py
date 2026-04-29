"""Render an extraction run as a single HTML for side-by-side PDF comparison.

Reads ``<run>/content_list.json`` and emits ``<run>/review.html``. All content
is taken verbatim from the extractor sidecars: text/markdown/html/latex stay
unchanged, image paths are relative so the file works inside the run folder,
nothing is invented. Elements appear in reading order, grouped by page.

Run from the project root:

    python scripts/build_review_view.py outputs/jmmp-09-00199-v2
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       color: #1a1a1a; background: #fafafa; line-height: 1.55; }
header.doc { padding: 24px 32px; background: #1f3a5f; color: #fff; }
header.doc h1 { margin: 0 0 6px 0; font-size: 20px; font-weight: 600; }
header.doc .docmeta { font-size: 12px; opacity: 0.8; }
main { max-width: 980px; margin: 0 auto; padding: 24px 32px 80px; }
.page-marker { margin: 32px 0 12px; padding: 6px 12px; background: #1f3a5f; color: #fff;
               font-size: 12px; letter-spacing: 0.05em; border-radius: 3px;
               display: flex; justify-content: space-between; align-items: center; }
.page-marker a { color: #cfe0f5; text-decoration: none; }
.page-marker a:hover { text-decoration: underline; }
section.el { margin: 14px 0; padding: 10px 14px; background: #fff;
             border-left: 3px solid #d0d7de; border-radius: 2px; }
section.el .meta { font-size: 11px; color: #6a737d; font-family: ui-monospace, monospace;
                   margin-bottom: 6px; }
section.heading { border-left-color: #1f3a5f; }
section.heading h2 { margin: 4px 0; font-size: 18px; font-weight: 600; }
section.text p { margin: 4px 0; white-space: pre-wrap; }
section.table { border-left-color: #c61f37; }
section.table .table-html table { border-collapse: collapse; margin: 6px 0; }
section.table .table-html td, section.table .table-html th {
    border: 1px solid #d0d7de; padding: 4px 8px; vertical-align: top; }
section.table pre.table-md { background: #f6f8fa; padding: 8px; overflow-x: auto;
                              font-size: 12px; border-radius: 3px; }
section.figure, section.diagram, section.technical_drawing { border-left-color: #d2992f; }
section.formula { border-left-color: #6f42c1; background: #faf8ff; }
section.formula .formula-block { font-size: 16px; padding: 8px 0; overflow-x: auto; }
img.crop { display: block; max-width: 100%; height: auto; margin: 6px 0;
           border: 1px solid #e1e4e8; background: #fff; }
img.formula-img { max-height: 80px; width: auto; }
.caption { font-size: 13px; color: #444; font-style: italic; margin: 4px 0; }
.description { font-size: 13px; color: #555; margin: 4px 0;
               border-left: 2px solid #e1e4e8; padding-left: 8px; }
"""

MATHJAX = (
    '<script>window.MathJax = { tex: { inlineMath: [["$","$"],["\\\\(","\\\\)"]] } };</script>'
    '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>'
)


def _label(el: dict) -> str:
    return (
        f"<div class='meta'>p.{el['page']} &middot; {el['type']} &middot; "
        f"{el['element_id']} &middot; conf {el['confidence']:.2f} &middot; "
        f"{html.escape(el['extractor'])}</div>"
    )


def render_element(el: dict) -> str:
    t = el["type"]
    c = el.get("content", {})
    parts: list[str] = [_label(el)]

    if t == "heading":
        parts.append(f"<h2>{html.escape(c.get('text', ''))}</h2>")

    elif t == "text":
        parts.append(f"<p>{html.escape(c.get('text', ''))}</p>")

    elif t == "table":
        if c.get("caption"):
            parts.append(f"<div class='caption'>{html.escape(c['caption'])}</div>")
        if c.get("html"):
            parts.append(f"<div class='table-html'>{c['html']}</div>")
        elif c.get("markdown"):
            parts.append(f"<pre class='table-md'>{html.escape(c['markdown'])}</pre>")
        if c.get("image_path"):
            parts.append(
                f"<img class='crop' loading='lazy' src='{html.escape(c['image_path'])}' alt='table crop'>"
            )

    elif t in ("figure", "diagram", "technical_drawing"):
        if c.get("image_path"):
            parts.append(
                f"<img class='crop' loading='lazy' src='{html.escape(c['image_path'])}' alt='{t}'>"
            )
        if c.get("caption"):
            parts.append(f"<div class='caption'>{html.escape(c['caption'])}</div>")
        if c.get("description"):
            parts.append(f"<div class='description'>{html.escape(c['description'])}</div>")

    elif t == "formula":
        if c.get("latex"):
            parts.append(f"<div class='formula-block'>$$ {c['latex']} $$</div>")
        if c.get("image_path"):
            parts.append(
                f"<img class='crop formula-img' loading='lazy' "
                f"src='{html.escape(c['image_path'])}' alt='formula'>"
            )
        if c.get("text") and not c.get("latex"):
            parts.append(f"<p>{html.escape(c['text'])}</p>")

    else:
        # Unknown type: dump fields as a plain block so nothing is silently lost.
        parts.append(f"<pre>{html.escape(json.dumps(c, indent=2, ensure_ascii=False))}</pre>")

    return f"<section class='el {t}' id='el_{el['element_id']}'>{''.join(parts)}</section>"


def render(run_dir: Path) -> str:
    data = json.loads((run_dir / "content_list.json").read_text(encoding="utf-8"))
    elements = data["elements"]

    by_page: dict[int, list[dict]] = defaultdict(list)
    for e in elements:
        by_page[e["page"]].append(e)

    out: list[str] = []
    out.append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    out.append(f'<title>Review &middot; {html.escape(data["source_file"])}</title>')
    out.append(f"<style>{CSS}</style>{MATHJAX}</head><body>")
    out.append("<header class='doc'>")
    out.append(f"<h1>{html.escape(data['source_file'])}</h1>")
    out.append(
        f"<div class='docmeta'>doc_id {data['doc_id']} &middot; "
        f"{data['total_pages']} pages &middot; "
        f"segmenter {html.escape(data['segmentation_tool'])} &middot; "
        f"{len(elements)} elements &middot; "
        f"schema {data['schema_version']}</div>"
    )
    out.append("</header><main>")

    for page_num in sorted(by_page):
        page_img = next(
            (p["image_path"] for p in data.get("pages", []) if p["page"] == page_num),
            None,
        )
        link = (
            f"<a href='{html.escape(page_img)}' target='_blank'>open page.png &rarr;</a>"
            if page_img
            else ""
        )
        out.append(
            f"<div class='page-marker' id='page_{page_num}'>"
            f"<span>Page {page_num} &middot; {len(by_page[page_num])} elements</span>"
            f"{link}</div>"
        )
        for e in by_page[page_num]:
            out.append(render_element(e))

    out.append("</main></body></html>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="Path to outputs/<run>/")
    parser.add_argument("--out", type=Path, default=None, help="Default: <run>/review.html")
    args = parser.parse_args()

    if not (args.run_dir / "content_list.json").is_file():
        raise SystemExit(f"no content_list.json in {args.run_dir}")

    out = args.out or args.run_dir / "review.html"
    out.write_text(render(args.run_dir), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
