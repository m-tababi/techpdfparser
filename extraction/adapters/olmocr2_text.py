"""olmOCR-2 text extractor.

Runs the allenai/olmOCR-2 Vision2Seq model over a cropped region image
and returns markdown text. The pipeline crops the region from the
rendered page before calling this adapter — the crop carries exactly
the text block, heading, or caption that the segmenter detected.

Model: allenai/olmOCR-2-7B-1025
Requires: pip install transformers torch olmocr
"""
from __future__ import annotations

import base64
import re
import warnings
from io import BytesIO
from typing import Any

from PIL import Image as PILImage

from .._runtime import is_cuda_oom, release_runtime_resources
from ..models import ElementContent
from ..registry import register_text_extractor

_OLMOCR_PROMPT = (
    "Attached is a cropped region from one page of a technical document. "
    "Return the plain text representation of this region as if you were "
    "reading it naturally.\n"
    "Convert equations to LaTeX and tables to HTML.\n"
    "Do not speculate about content outside the crop."
)
_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)
_TARGET_LONGEST_DIM = 1288


@register_text_extractor("olmocr2")
class OlmOCR2TextExtractor:
    TOOL_NAME = "olmocr2"

    def __init__(
        self,
        model_name: str = "allenai/olmOCR-2-7B-1025",
        processor_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
    ) -> None:
        self._model_name = model_name
        self._processor_name = processor_name
        self._device = device
        self._runtime_device = device
        self._model: Any = None
        self._processor: Any = None

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._processor_name)
            self._load_model(AutoModelForVision2Seq)
        except ImportError as exc:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            ) from exc

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

    def extract(self, page_image: Any, page_number: int) -> ElementContent:
        self._load()
        text = self._run_ocr(page_image)
        return ElementContent(text=text)

    def _run_ocr(self, image: Any) -> str:
        import torch

        image = self._prepare_image(image)
        image_base64 = self._image_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _OLMOCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self._runtime_device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
                num_return_sequences=1,
            )
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = self._processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()
        return self._strip_front_matter(text_output)

    @staticmethod
    def _prepare_image(image: Any) -> Any:
        longest_dim = max(image.size)
        if longest_dim == _TARGET_LONGEST_DIM:
            return image
        scale = _TARGET_LONGEST_DIM / longest_dim
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        return image.resize(new_size, resample=PILImage.Resampling.LANCZOS)

    @staticmethod
    def _image_to_base64(image: Any) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _strip_front_matter(text: str) -> str:
        return _FRONT_MATTER_RE.sub("", text, count=1).strip()
