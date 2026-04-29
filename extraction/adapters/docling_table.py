"""Docling TableFormer table extractor.

Standalone use of IBM's TableFormer structure-recognition model on a
cropped table image. TableFormer outputs the cell grid (row/col spans,
header flags, cell bboxes) but does NOT do OCR — cell text is sourced
from a Tesseract pass over the same crop and matched into cells by
bbox containment.

Model weights: ``ds4sd/docling-models`` (HF), ``model_artifacts/tableformer/{fast,accurate}``
Requires: ``pip install docling-ibm-models pytesseract``
System binary: ``tesseract-ocr`` (e.g. ``apt install tesseract-ocr tesseract-ocr-deu``)
"""
from __future__ import annotations

from typing import Any

from PIL.Image import Image

from ..models import ElementContent
from ..registry import register_table_extractor

_DEFAULT_REPO = "ds4sd/docling-models"
_DEFAULT_REVISION = "v2.1.0"
_DEFAULT_VARIANT = "fast"  # alternative: "accurate"
_DEFAULT_LANGS = "deu+eng"


@register_table_extractor("docling_table")
class DoclingTableExtractor:
    TOOL_NAME = "docling_table"

    def __init__(
        self,
        repo_id: str = _DEFAULT_REPO,
        revision: str = _DEFAULT_REVISION,
        variant: str = _DEFAULT_VARIANT,
        device: str = "cuda",
        ocr_langs: str = _DEFAULT_LANGS,
    ) -> None:
        self._repo_id = repo_id
        self._revision = revision
        self._variant = variant
        self._device = device
        self._ocr_langs = ocr_langs
        self._predictor: Any = None

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def _load(self) -> None:
        if self._predictor is not None:
            return
        try:
            from docling_ibm_models.tableformer.data_management.tf_predictor import (
                TFPredictor,
            )
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "docling_ibm_models not installed. "
                "Run: pip install docling-ibm-models"
            ) from exc

        weights_dir = snapshot_download(repo_id=self._repo_id, revision=self._revision)
        config = _build_predictor_config(weights_dir, self._variant)
        self._predictor = TFPredictor(config, device=self._device, num_threads=4)

    def unload(self) -> None:
        self._predictor = None

    def extract(self, region_image: Image, page_number: int) -> ElementContent:
        self._load()
        assert self._predictor is not None

        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "pytesseract not installed. Run: pip install pytesseract "
                "(plus system binary, e.g. `apt install tesseract-ocr`)"
            ) from exc

        import numpy as np

        rgb = region_image.convert("RGB")
        width, height = rgb.size
        tokens = _ocr_tokens(rgb, pytesseract, self._ocr_langs)

        iocr_page = {
            "image": np.array(rgb),
            "tokens": tokens,
            "width": width,
            "height": height,
        }
        table_bbox = [0.0, 0.0, float(width), float(height)]

        try:
            results = self._predictor.multi_table_predict(
                iocr_page,
                [table_bbox],
                do_matching=True,
                correct_overlapping_cells=False,
                sort_row_col_indexes=True,
            )
        except Exception as exc:  # noqa: BLE001 — adapter degrades gracefully
            return ElementContent(text=f"[docling_table extraction failed: {exc}]")

        if not results:
            return ElementContent()

        cells = results[0].get("tf_responses") or []
        html = _cells_to_html(cells, tokens)
        text = _html_to_plain(html)
        return ElementContent(html=html or None, text=text or None)


def _build_predictor_config(weights_dir: str, variant: str) -> dict[str, Any]:
    """Load tm_config.json shipped with the snapshot, point save_dir at it.

    The TFPredictor requires many keys (dataset_wordmap, predict.padding,
    predict.pdf_cell_iou_thres, ...) that the snapshot's tm_config.json
    already provides. Hand-crafting the dict drifts whenever upstream adds
    a new required key; reusing the shipped config is the robust path.
    """
    import json
    import os

    save_dir = os.path.join(weights_dir, "model_artifacts", "tableformer", variant)
    with open(os.path.join(save_dir, "tm_config.json")) as f:
        config: dict[str, Any] = json.load(f)
    config["model"]["save_dir"] = save_dir
    return config


def _ocr_tokens(image: Image, pytesseract: Any, langs: str) -> list[dict[str, Any]]:
    data = pytesseract.image_to_data(
        image, lang=langs, output_type=pytesseract.Output.DICT
    )
    tokens: list[dict[str, Any]] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        x, y, w, h = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        tokens.append(
            {
                "id": i,
                "text": text,
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
            }
        )
    return tokens


def _cells_to_html(
    cells: list[dict[str, Any]], tokens: list[dict[str, Any]]
) -> str:
    """Build a single ``<table>`` from TableFormer cell dicts.

    Honors row_span/col_span and uses ``<th>`` for header cells. Prefers
    the cell's ``start_row_offset_idx`` / ``start_col_offset_idx`` fields
    when present; otherwise derives row/column ordering from the cell
    bbox centers (the high-level ``multi_table_predict`` wrapper does not
    always expose offset indices).
    """
    if not cells:
        return ""

    rows = _group_cells_into_rows(cells)
    parts = ["<table>"]
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            tag = "th" if (cell.get("column_header") or cell.get("row_header")) else "td"
            attrs: list[str] = []
            rs = int(cell.get("row_span", 1))
            cs = int(cell.get("col_span", 1))
            if rs > 1:
                attrs.append(f'rowspan="{rs}"')
            if cs > 1:
                attrs.append(f'colspan="{cs}"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            text = _cell_text(cell, tokens)
            parts.append(f"<{tag}{attr_str}>{_html_escape(text)}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _group_cells_into_rows(
    cells: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Order cells into rows + ordered columns within each row."""
    has_offsets = any(
        int(c.get("start_row_offset_idx", 0)) != 0
        or int(c.get("start_col_offset_idx", 0)) != 0
        for c in cells
    )
    if has_offsets:
        by_row: dict[int, list[dict[str, Any]]] = {}
        for cell in cells:
            r = int(cell.get("start_row_offset_idx", 0))
            by_row.setdefault(r, []).append(cell)
        return [
            sorted(by_row[r], key=lambda c: int(c.get("start_col_offset_idx", 0)))
            for r in sorted(by_row)
        ]

    # Fallback: cluster cells by bbox y-center, then sort within row by x-center.
    enriched = sorted(
        ((_cell_y(c), _cell_x(c), c) for c in cells), key=lambda t: (t[0], t[1])
    )
    heights = [h for h in (_cell_h(c) for _, _, c in enriched) if h > 0]
    tol = (sorted(heights)[len(heights) // 2] * 0.5) if heights else 5.0
    rows_out: list[list[dict[str, Any]]] = []
    current: list[tuple[float, dict[str, Any]]] = []
    current_y: float | None = None
    for y, x, cell in enriched:
        if current_y is not None and abs(y - current_y) > tol:
            current.sort(key=lambda t: t[0])
            rows_out.append([c for _, c in current])
            current = []
            current_y = None
        current.append((x, cell))
        current_y = y if current_y is None else current_y
    if current:
        current.sort(key=lambda t: t[0])
        rows_out.append([c for _, c in current])
    return rows_out


def _cell_y(cell: dict[str, Any]) -> float:
    bb = cell.get("bbox") or {}
    if isinstance(bb, dict):
        return 0.5 * (float(bb.get("t", 0)) + float(bb.get("b", 0)))
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        return 0.5 * (float(bb[1]) + float(bb[3]))
    return 0.0


def _cell_x(cell: dict[str, Any]) -> float:
    bb = cell.get("bbox") or {}
    if isinstance(bb, dict):
        return 0.5 * (float(bb.get("l", 0)) + float(bb.get("r", 0)))
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        return 0.5 * (float(bb[0]) + float(bb[2]))
    return 0.0


def _cell_h(cell: dict[str, Any]) -> float:
    bb = cell.get("bbox") or {}
    if isinstance(bb, dict):
        return float(bb.get("b", 0)) - float(bb.get("t", 0))
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        return float(bb[3]) - float(bb[1])
    return 0.0


def _cell_text(cell: dict[str, Any], tokens: list[dict[str, Any]]) -> str:
    """Return text for a cell, preferring matched text over bbox-intersection lookup."""
    matched = cell.get("text") or cell.get("bbox", {}).get("token")
    if isinstance(matched, str) and matched.strip():
        return matched.strip()

    bbox = cell.get("bbox") or {}
    if isinstance(bbox, dict):
        cell_box = [
            float(bbox.get("l", 0)),
            float(bbox.get("t", 0)),
            float(bbox.get("r", 0)),
            float(bbox.get("b", 0)),
        ]
    else:
        cell_box = [float(v) for v in bbox][:4]
    if cell_box[2] <= cell_box[0] or cell_box[3] <= cell_box[1]:
        return ""

    pieces = [
        tok["text"]
        for tok in tokens
        if _bbox_inside(tok["bbox"], cell_box)
    ]
    return " ".join(pieces).strip()


def _bbox_inside(inner: list[float], outer: list[float], pad: float = 2.0) -> bool:
    cx = 0.5 * (inner[0] + inner[2])
    cy = 0.5 * (inner[1] + inner[3])
    return (
        outer[0] - pad <= cx <= outer[2] + pad
        and outer[1] - pad <= cy <= outer[3] + pad
    )


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _html_to_plain(html: str) -> str:
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
