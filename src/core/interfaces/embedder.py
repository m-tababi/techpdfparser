from __future__ import annotations

from typing import Protocol

from ..indexing import VectorSchema


class TextEmbedder(Protocol):
    """Generates dense text embeddings for chunks, tables, formulas, and figures.

    Swap adapters to compare embedding models (BGE-M3, E5, GTE, etc.)
    without touching any pipeline or storage code.
    """

    @property
    def tool_name(self) -> str: ...

    @property
    def tool_version(self) -> str: ...

    @property
    def embedding_dim(self) -> int:
        """Output vector dimension. Used to configure the vector DB collection."""
        ...

    @property
    def vector_schema(self) -> VectorSchema:
        """Schema used by the vector DB for this embedder."""
        ...

    @property
    def adapter_signature(self) -> str:
        """Deterministic signature for namespaceing and manifests."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Returns one vector per input text."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        ...
