from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.registry import register_figure_descriptor

if TYPE_CHECKING:
    from PIL.Image import Image

_DESCRIBE_PROMPT = (
    "Describe this figure from a technical document concisely. "
    "Identify the visualization type (chart, diagram, graph, schematic, etc.), "
    "what data or concept it shows, and any key values or trends visible. "
    "Be specific and technical. Two to four sentences maximum."
)


@register_figure_descriptor("qwen25vl")
class Qwen25VLDescriptor:
    """Figure descriptor using Qwen2.5-VL-7B-Instruct.

    Generates concise textual descriptions of charts, diagrams, and figures.
    The description is embedded as text for semantic retrieval of visual content.

    Replace with LLaVA, InternVL, or any other VLM by registering under a
    different name and updating `pipelines.structured.figure_descriptor` in config.

    Model: Qwen/Qwen2.5-VL-7B-Instruct
    Requires: pip install transformers torch
    """

    TOOL_NAME = "qwen25vl"
    TOOL_VERSION = "7b"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self._model_name, device_map=self._device
            )
            self._processor = AutoProcessor.from_pretrained(self._model_name)
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    @property
    def tool_version(self) -> str:
        return self.TOOL_VERSION

    def describe(self, image: Image) -> str:
        """Generate a technical description of a figure image."""
        self._load()
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _DESCRIBE_PROMPT},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=256)

        # Decode only the newly generated tokens (skip prompt)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()
