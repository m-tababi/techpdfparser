from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.registry import register_figure_descriptor

if TYPE_CHECKING:
    from PIL.Image import Image

# Same prompt as qwen25vl.py so descriptions are semantically comparable
# when A/B testing the two adapters on the same figures.
_DESCRIBE_PROMPT = (
    "Describe this figure from a technical document concisely. "
    "Identify the visualization type (chart, diagram, graph, schematic, etc.), "
    "what data or concept it shows, and any key values or trends visible. "
    "Be specific and technical. Two to four sentences maximum."
)


@register_figure_descriptor("moondream")
class MoondreamDescriptor:
    """Figure descriptor using Moondream2 (~2B params).

    CPU-viable alternative to Qwen2.5-VL-7B for testing on AMD/CPU hardware.
    Inference is slow on CPU (~5-15s per image) but functional.

    revision is pinned because the moondream2 HuggingFace repo updates weights
    in-place without changing the model name — pinning ensures reproducibility.

    Requires: pip install transformers einops
    """

    TOOL_NAME = "moondream"
    TOOL_VERSION = "2"

    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        revision: str = "2025-01-09",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._revision = revision
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, revision=self._revision
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name, revision=self._revision, trust_remote_code=True
            )
            self._model.to(self._device)
            self._model.eval()
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers einops"
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
        encoded = self._model.encode_image(image)
        return self._model.answer_question(encoded, _DESCRIBE_PROMPT, self._tokenizer)
