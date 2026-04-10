from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.registry import register_visual_embedder

if TYPE_CHECKING:
    from PIL.Image import Image


@register_visual_embedder("colqwen25")
class ColQwen25Embedder:
    """Visual embedder using ColQwen2.5 (late-interaction multi-vector model).

    ColQwen2.5 produces one 128-dim vector per image patch. Retrieval uses
    MaxSim scoring across all patch pairs, which outperforms single-vector
    CLIP-style models for fine-grained document retrieval.

    Replace with ColPali by registering an adapter under "colpali" and
    changing `pipelines.visual.embedder` in the config.

    Model: vidore/colqwen2.5-v0.2
    Requires: pip install colpali-engine torch
    """

    TOOL_NAME = "colqwen25"
    TOOL_VERSION = "0.2"
    # Per-patch vector dimension for colqwen2.5-v0.2
    EMBEDDING_DIM = 128

    def __init__(
        self,
        model_name: str = "vidore/colqwen2.5-v0.2",
        device: str = "cuda",
        batch_size: int = 4,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Lazy-load on first use to avoid slow startup when not all pipelines run."""
        if self._model is not None:
            return
        try:
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

            self._model = ColQwen2_5.from_pretrained(self._model_name).to(self._device)
            self._processor = ColQwen2_5_Processor.from_pretrained(self._model_name)
            self._model.eval()
        except ImportError:
            raise ImportError(
                "colpali-engine not installed. Run: pip install colpali-engine"
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
        return True

    def embed_page(self, image: Image) -> list[list[float]]:
        """Embed a page image into N patch vectors (N depends on image size)."""
        self._load()
        import torch

        batch = self._processor.process_images([image]).to(self._device)
        with torch.no_grad():
            embeddings = self._model(**batch)
        # Shape: (1, num_patches, dim) → (num_patches, dim)
        return embeddings[0].cpu().float().tolist()

    def embed_query(self, query: str) -> list[list[float]]:
        """Embed a query into token vectors for MaxSim scoring against pages."""
        self._load()
        import torch

        batch = self._processor.process_queries([query]).to(self._device)
        with torch.no_grad():
            embeddings = self._model(**batch)
        return embeddings[0].cpu().float().tolist()
