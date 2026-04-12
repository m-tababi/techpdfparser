from __future__ import annotations

from ...core.indexing import VectorSchema, build_adapter_signature
from ...core.registry import register_text_embedder


@register_text_embedder("minilm")
class MiniLMEmbedder:
    """Dense text embedder using all-MiniLM-L6-v2 via sentence-transformers.

    Lightweight CPU-native alternative to BGE-M3. Produces 384-dim vectors
    instead of 1024-dim, so collections are not interchangeable between the
    two adapters — Qdrant collection schema is determined at first run.

    Replace with a larger model (e.g. all-mpnet-base-v2 at 768-dim) by
    registering a new adapter; only the model_name and EMBEDDING_DIM change.

    Requires: pip install sentence-transformers
    """

    TOOL_NAME = "minilm"
    TOOL_VERSION = "L6-v2"
    EMBEDDING_DIM = 384

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
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
    def model_name(self) -> str:
        return self._model_name

    @property
    def vector_schema(self) -> VectorSchema:
        return VectorSchema(dim=self.embedding_dim, distance="cosine", multi_vector=False)

    @property
    def adapter_signature(self) -> str:
        return build_adapter_signature(
            tool_name=self.tool_name,
            model_name=self.model_name,
            schema=self.vector_schema,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in batches. Returns one 384-dim vector per text."""
        self._load()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._encode_batch(batch))
        return results

    def embed_query(self, query: str) -> list[float]:
        self._load()
        vecs = self._encode_batch([query])
        return vecs[0]

    def _encode_batch(self, batch: list[str]) -> list[list[float]]:
        import numpy as np

        arr = self._model.encode(batch, convert_to_numpy=True)
        # encode() may return a 1-D array when batch has one element
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr.tolist()
