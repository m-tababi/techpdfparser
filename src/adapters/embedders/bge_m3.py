from __future__ import annotations

from ...core.registry import register_text_embedder


@register_text_embedder("bge_m3")
class BGEM3Embedder:
    """Dense text embedder using BGE-M3.

    BGE-M3 supports up to 8192-token context and produces 1024-dim vectors.
    It works well for multilingual technical content and long passages.

    Replace with E5-large, GTE, or another text embedder by registering
    under a different name and updating `pipelines.text.embedder` in config.

    Model: BAAI/bge-m3
    Requires: pip install FlagEmbedding
    """

    TOOL_NAME = "bge_m3"
    TOOL_VERSION = "1.0"
    EMBEDDING_DIM = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel

            # fp16 halves memory; safe for inference
            self._model = BGEM3FlagModel(self._model_name, use_fp16=True)
        except ImportError:
            raise ImportError(
                "FlagEmbedding not installed. Run: pip install FlagEmbedding"
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in batches. Returns one vector per text."""
        self._load()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            output = self._model.encode(batch, batch_size=len(batch))
            results.extend(output["dense_vecs"].tolist())
        return results

    def embed_query(self, query: str) -> list[float]:
        self._load()
        output = self._model.encode([query])
        return output["dense_vecs"][0].tolist()
