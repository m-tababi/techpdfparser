from __future__ import annotations

from typing import Protocol

from PIL.Image import Image

from ..indexing import VectorSchema


class VisualEmbedder(Protocol):
    """Generates visual embeddings for page images.

    Returns list[list[float]] in all cases:
    - Late-interaction models (ColQwen2.5, ColPali): N patch vectors per image.
    - Single-vector models (CLIP, SigLIP): one vector wrapped in an outer list.

    The IndexWriter checks `is_multi_vector` to choose the right storage format.
    """

    @property
    def tool_name(self) -> str: ...

    @property
    def tool_version(self) -> str: ...

    @property
    def embedding_dim(self) -> int:
        """Dimension of each individual vector."""
        ...

    @property
    def is_multi_vector(self) -> bool:
        """True for late-interaction models like ColQwen2.5 / ColPali."""
        ...

    @property
    def vector_schema(self) -> VectorSchema:
        """Schema used by the vector DB for this embedder."""
        ...

    @property
    def adapter_signature(self) -> str:
        """Deterministic signature for namespaceing and manifests."""
        ...

    def embed_page(self, image: Image) -> list[list[float]]:
        """Generate visual embedding(s) for one page image."""
        ...

    def embed_query(self, query: str) -> list[list[float]]:
        """Generate query embedding(s) in the same space as page embeddings."""
        ...
