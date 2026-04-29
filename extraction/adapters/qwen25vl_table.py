"""Qwen2.5-VL table extractor.

Reuses the same model class as the figure descriptor, but with a
table-specific prompt that forces a single ``<table>`` HTML block as
output. Suitable when the segmenter has already isolated a table region
crop — the VLM does both structure recognition and OCR in one shot.

Model: ``Qwen/Qwen2.5-VL-7B-Instruct``
Requires: ``pip install transformers torch``
"""
from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any

from .._runtime import is_cuda_oom, release_runtime_resources
from ..models import ElementContent
from ..registry import register_table_extractor

if TYPE_CHECKING:
    from PIL.Image import Image

_TABLE_PROMPT = (
    "This is a cropped table from a technical document. Convert it to a "
    "single HTML table. Use <th> for header cells (column headers and row "
    "headers) and <td> for data cells. Use rowspan and colspan attributes "
    "when cells span multiple rows or columns. Preserve units, decimal "
    "separators, and symbols exactly as shown. If a cell is unreadable or "
    "empty, leave it empty (no guessing, no placeholder text). "
    "Return ONLY the HTML <table>...</table> with no explanation, no "
    "Markdown fence, no prose."
)

_FENCE_RE = re.compile(r"^```(?:html)?\s*\n?|\n?```$", re.IGNORECASE)
_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)


@register_table_extractor("qwen25vl_table")
class Qwen25VLTableExtractor:
    TOOL_NAME = "qwen25vl_table"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 2048,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._runtime_device = device
        self._max_new_tokens = int(max_new_tokens)
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
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._load_model(Qwen2_5_VLForConditionalGeneration)

    def _load_model(self, model_cls: Any) -> None:
        try:
            self._model = model_cls.from_pretrained(
                self._model_name,
                device_map=self._runtime_device,
                torch_dtype="auto",
            )
        except Exception as exc:
            if self._runtime_device == "cpu" or not is_cuda_oom(exc):
                raise
            warnings.warn(
                f"{self.TOOL_NAME} ran out of GPU memory; retrying on CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            processor = self._processor
            self._model = None
            release_runtime_resources()
            self._runtime_device = "cpu"
            self._processor = processor
            self._model = model_cls.from_pretrained(
                self._model_name,
                device_map=self._runtime_device,
                torch_dtype="auto",
            )

    def unload(self) -> None:
        self._model = None
        self._processor = None

    def extract(self, region_image: "Image", page_number: int) -> ElementContent:
        self._load()
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": region_image},
                    {"type": "text", "text": _TABLE_PROMPT},
                ],
            }
        ]
        text_in = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_in], images=[region_image], return_tensors="pt"
        ).to(self._runtime_device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens
            )

        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        raw = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
        html = _extract_table_html(raw)
        text = _html_to_plain(html)
        return ElementContent(html=html or None, text=text or None)


def _extract_table_html(raw: str) -> str:
    """Strip code fences and return the first ``<table>...</table>`` block."""
    if not raw:
        return ""
    cleaned = _FENCE_RE.sub("", raw).strip()
    match = _TABLE_RE.search(cleaned)
    if match:
        return match.group(0)
    # Model didn't wrap in <table> — treat whole output as table content if
    # it at least starts with a row tag, else return as-is for inspection.
    if cleaned.lstrip().lower().startswith("<tr"):
        return f"<table>{cleaned}</table>"
    return cleaned


def _html_to_plain(html: str) -> str:
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
