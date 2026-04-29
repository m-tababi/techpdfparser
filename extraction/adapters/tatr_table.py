"""Microsoft Table Transformer (TATR) v1.1 structure recognition.

Pure structure model: takes a cropped table image, returns rows,
columns, headers, and spanning cells. Cell text is sourced from a
Tesseract OCR pass over the crop and assigned to grid cells by
bbox-center containment.

Model: ``microsoft/table-transformer-structure-recognition-v1.1-all``
Requires: ``pip install transformers torch pytesseract``
System binary: ``tesseract-ocr`` (e.g. ``apt install tesseract-ocr tesseract-ocr-deu``)
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from .._runtime import is_cuda_oom, release_runtime_resources
from ..models import ElementContent
from ..registry import register_table_extractor

if TYPE_CHECKING:
    from PIL.Image import Image

_DEFAULT_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"
_DEFAULT_DETECTOR_PROCESSOR = "microsoft/table-transformer-detection"
_DEFAULT_LANGS = "deu+eng"
_SCORE_THRESHOLD = 0.5

# TATR structure-recognition class names
_LBL_TABLE = "table"
_LBL_COLUMN = "table column"
_LBL_ROW = "table row"
_LBL_COLUMN_HEADER = "table column header"
_LBL_ROW_HEADER = "table projected row header"
_LBL_SPANNING = "table spanning cell"


@register_table_extractor("tatr")
class TATRTableExtractor:
    TOOL_NAME = "tatr"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        processor_name: str = _DEFAULT_DETECTOR_PROCESSOR,
        device: str = "cuda",
        ocr_langs: str = _DEFAULT_LANGS,
        score_threshold: float = _SCORE_THRESHOLD,
    ) -> None:
        self._model_name = model_name
        self._processor_name = processor_name
        self._device = device
        self._runtime_device = device
        self._ocr_langs = ocr_langs
        self._score_threshold = float(score_threshold)
        self._model: Any = None
        self._processor: Any = None

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoImageProcessor,
                TableTransformerForObjectDetection,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            ) from exc

        self._processor = AutoImageProcessor.from_pretrained(self._processor_name)
        self._load_model(TableTransformerForObjectDetection)

    def _load_model(self, model_cls: Any) -> None:
        try:
            self._model = model_cls.from_pretrained(self._model_name).to(
                self._runtime_device
            )
        except Exception as exc:
            if self._runtime_device == "cpu" or not is_cuda_oom(exc):
                raise
            warnings.warn(
                f"{self.TOOL_NAME} ran out of GPU memory; retrying on CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            release_runtime_resources()
            self._runtime_device = "cpu"
            self._model = model_cls.from_pretrained(self._model_name).to("cpu")

    def unload(self) -> None:
        self._model = None
        self._processor = None

    def extract(self, region_image: "Image", page_number: int) -> ElementContent:
        self._load()
        assert self._model is not None and self._processor is not None
        try:
            import pytesseract
            import torch
        except ImportError as exc:
            raise ImportError(
                "pytesseract or torch missing. "
                "Run: pip install pytesseract torch"
            ) from exc

        rgb = region_image.convert("RGB")
        inputs = self._processor(images=rgb, return_tensors="pt").to(
            self._runtime_device
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([rgb.size[::-1]], device=self._runtime_device)
        results = self._processor.post_process_object_detection(
            outputs, threshold=self._score_threshold, target_sizes=target_sizes
        )[0]

        id2label = self._model.config.id2label
        predictions = _to_predictions(results, id2label)
        if not predictions:
            return ElementContent()

        tokens = _ocr_tokens(rgb, pytesseract, self._ocr_langs)
        html = _predictions_to_html(predictions, tokens)
        text = _html_to_plain(html)
        return ElementContent(html=html or None, text=text or None)


def _to_predictions(results: dict[str, Any], id2label: dict[int, str]) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels")
    if boxes is None or scores is None or labels is None:
        return preds
    for box, score, label in zip(
        boxes.tolist(), scores.tolist(), labels.tolist(), strict=False
    ):
        name = id2label.get(int(label))
        if name is None:
            continue
        preds.append({"label": name, "score": float(score), "bbox": [float(v) for v in box]})
    return preds


def _ocr_tokens(image: Any, pytesseract: Any, langs: str) -> list[dict[str, Any]]:
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
            {"text": text, "bbox": [float(x), float(y), float(x + w), float(y + h)]}
        )
    return tokens


def _predictions_to_html(
    predictions: list[dict[str, Any]], tokens: list[dict[str, Any]]
) -> str:
    """Build an HTML table from row/column/header/spanning predictions + tokens.

    Strategy:
    1. Sort row predictions by y, column predictions by x → grid coordinates.
    2. For each (row, column) pair, intersect bboxes → cell bbox.
    3. A cell is a header if its bbox center lies inside any column-header
       or row-header prediction.
    4. Spanning cells override individual (row, column) cells they cover.
    5. Cell text = concatenation of OCR tokens whose bbox center lies in the cell.
    """
    rows = sorted(
        [p for p in predictions if p["label"] == _LBL_ROW],
        key=lambda p: p["bbox"][1],
    )
    cols = sorted(
        [p for p in predictions if p["label"] == _LBL_COLUMN],
        key=lambda p: p["bbox"][0],
    )
    col_headers = [p for p in predictions if p["label"] == _LBL_COLUMN_HEADER]
    row_headers = [p for p in predictions if p["label"] == _LBL_ROW_HEADER]
    spans = [p for p in predictions if p["label"] == _LBL_SPANNING]

    if not rows or not cols:
        return ""

    # Mark cells covered by a spanning cell so we emit them once.
    consumed: set[tuple[int, int]] = set()
    span_anchors: dict[tuple[int, int], tuple[int, int]] = {}
    for span in spans:
        s_box = span["bbox"]
        covered = [
            (r, c)
            for r in range(len(rows))
            for c in range(len(cols))
            if _intersects(_cell_bbox(rows[r], cols[c]), s_box)
        ]
        if not covered:
            continue
        anchor = min(covered)
        rs = max(r for r, _ in covered) - anchor[0] + 1
        cs = max(c for _, c in covered) - anchor[1] + 1
        span_anchors[anchor] = (rs, cs)
        for rc in covered:
            if rc != anchor:
                consumed.add(rc)

    parts = ["<table>"]
    for r_idx, row in enumerate(rows):
        parts.append("<tr>")
        for c_idx, col in enumerate(cols):
            if (r_idx, c_idx) in consumed:
                continue
            cell_box = _cell_bbox(row, col)
            is_header = _is_header(cell_box, col_headers, row_headers)
            tag = "th" if is_header else "td"
            attrs: list[str] = []
            if (r_idx, c_idx) in span_anchors:
                rs, cs = span_anchors[(r_idx, c_idx)]
                # When a span covers the cell, use the span's bbox for OCR
                # so we capture text that crosses sub-cell boundaries.
                for span in spans:
                    if _intersects(cell_box, span["bbox"]):
                        cell_box = span["bbox"]
                        break
                if rs > 1:
                    attrs.append(f'rowspan="{rs}"')
                if cs > 1:
                    attrs.append(f'colspan="{cs}"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            text = " ".join(
                tok["text"] for tok in tokens if _bbox_inside(tok["bbox"], cell_box)
            ).strip()
            parts.append(f"<{tag}{attr_str}>{_html_escape(text)}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _cell_bbox(row: dict[str, Any], col: dict[str, Any]) -> list[float]:
    rb = row["bbox"]
    cb = col["bbox"]
    return [
        max(rb[0], cb[0]),
        max(rb[1], cb[1]),
        min(rb[2], cb[2]),
        min(rb[3], cb[3]),
    ]


def _is_header(
    cell_box: list[float],
    col_headers: list[dict[str, Any]],
    row_headers: list[dict[str, Any]],
) -> bool:
    cx = 0.5 * (cell_box[0] + cell_box[2])
    cy = 0.5 * (cell_box[1] + cell_box[3])
    for header in col_headers + row_headers:
        b = header["bbox"]
        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
            return True
    return False


def _intersects(a: list[float], b: list[float]) -> bool:
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return False
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


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
