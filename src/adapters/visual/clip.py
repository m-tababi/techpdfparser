from __future__ import annotations

from typing import TYPE_CHECKING

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

    def embed_page(self, image: Image) -> list[list[float]]:
        """Embed a page image into a single 512-dim vector (wrapped in outer list)."""
        self._load()
        return [self._encode_image(image)]

    def embed_query(self, query: str) -> list[list[float]]:
        """Embed a query string into a single 512-dim vector (wrapped in outer list)."""
        self._load()
        return [self._encode_text(query)]

    def _encode_image(self, image: Image) -> list[float]:
        import torch

        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().float().tolist()

    def _encode_text(self, text: str) -> list[float]:
        import torch

        inputs = self._processor(text=text, return_tensors="pt", padding=True).to(
            self._device
        )
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().float().tolist()
