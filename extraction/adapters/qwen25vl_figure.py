"""Qwen2.5-VL figure descriptor.

Produces a short technical description of a figure/diagram crop for
downstream embedding.

Model: Qwen/Qwen2.5-VL-7B-Instruct
Requires: pip install transformers torch
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from .._runtime import is_cuda_oom, release_runtime_resources
from ..registry import register_figure_descriptor

if TYPE_CHECKING:
    from PIL.Image import Image

_DESCRIBE_PROMPT_BASE = (
    "Describe this figure from a technical document concisely. "
    "Identify the visualization type (chart, diagram, graph, schematic, etc.), "
    "what data or concept it shows, and any key values or trends visible. "
    "Be specific and technical. Two to four sentences maximum. "
    "If labels, axes, units, or the subject are unclear or not readable, say "
    "'unclear' or 'not readable' instead of guessing. Do not infer a domain, "
    "algorithm, or meaning that is not visible in the crop or stated in the "
    "provided source context."
)


def _build_prompt(caption: str | None) -> str:
    """Return the Qwen user-text, with caption grounding when non-empty."""
    if caption is None or not caption.strip():
        return _DESCRIBE_PROMPT_BASE
    # Concatenation (not .format) — der Caption-Text darf `{}` enthalten.
    return (
        _DESCRIBE_PROMPT_BASE
        + "\n\nThe figure has this caption from the source document:\n"
        + "  " + caption.strip() + "\n\n"
        + "Use the caption as grounding context. Do not invent any fact "
        + "that is not supported by either the image or the caption."
    )


@register_figure_descriptor("qwen25vl")
class Qwen25VLFigureDescriptor:
    TOOL_NAME = "qwen25vl"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
    ) -> None:
        self._model_name = model_name
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
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )

            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._load_model(Qwen2_5_VLForConditionalGeneration)
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

    def describe(self, image: Image, caption: str | None = None) -> str:
        self._load()
        import torch

        prompt_text = _build_prompt(caption)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self._runtime_device)

        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=256)

        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()
