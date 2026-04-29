from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.indexing import VectorSchema, build_adapter_signature
from ...core.registry import register_visual_embedder

if TYPE_CHECKING:
    from PIL.Image import Image


@register_visual_embedder("clip")
class CLIPEmbedder:
    """Visual embedder using CLIP (openai/clip-vit-base-patch32).

    CPU-native alternative to ColQwen2.5. Produces a single 512-dim vector
    per image rather than multiple patch vectors, so retrieval uses cosine
    similarity instead of MaxSim.

    is_multi_vector is True here because VisualPage.embedding is always
    list[list[float]], and the Qdrant writer passes it directly. Wrapping
    the single vector in [[v0…v511]] with a MaxSim collection is equivalent
    to cosine similarity — no pipeline or writer changes needed.

    Requires: pip install transformers
    torch CPU wheel: pip install torch --index-url https://download.pytorch.org/whl/cpu
    """

    TOOL_NAME = "clip"
    TOOL_VERSION = "vit-base-patch32"
    EMBEDDING_DIM = 512

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, CLIPModel

            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = CLIPModel.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()
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

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    @property
    def is_multi_vector(self) -> bool:
        # Single vector wrapped in outer list — see class docstring for why.
        return True

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vector_schema(self) -> VectorSchema:
        return VectorSchema(
            dim=self.embedding_dim,
            distance="cosine",
            multi_vector=self.is_multi_vector,
        )

    @property
    def adapter_signature(self) -> str:
        return build_adapter_signature(
            tool_name=self.tool_name,
            model_name=self.model_name,
            schema=self.vector_schema,
        )

    def embed_page(self, image: Image) -> list[list[float]]:
        """Embed a page image into a single 512-dim vector (wrapped in outer list)."""
        self._load()
        return [self._encode_image(image)]

    def embed_query(self, query: str) -> list[list[float]]:
        """Embed a query string into a single 512-dim vector (wrapped in outer list)."""
        self._load()
        return [self._encode_text(query)]

    @staticmethod
    def _to_tensor(output: object) -> "torch.Tensor":
        import torch

        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, (tuple, list)):
            return output[0]
        raise TypeError(f"Unexpected CLIP output type: {type(output)}")

    def _encode_image(self, image: Image) -> list[float]:
        import torch

        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            get_image_features = getattr(self._model, "get_image_features", None)
            if callable(get_image_features):
                features = self._to_tensor(
                    get_image_features(pixel_values=inputs["pixel_values"])
                )
            else:
                vision_out = self._model.vision_model(pixel_values=inputs["pixel_values"])
                features = self._model.visual_projection(vision_out.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().float().tolist()

    def _encode_text(self, text: str) -> list[float]:
        import torch

        inputs = self._processor(text=text, return_tensors="pt", padding=True).to(
            self._device
        )
        with torch.no_grad():
            get_text_features = getattr(self._model, "get_text_features", None)
            if callable(get_text_features):
                features = self._to_tensor(
                    get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
                )
            else:
                text_out = self._model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                features = self._model.text_projection(text_out.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().float().tolist()
