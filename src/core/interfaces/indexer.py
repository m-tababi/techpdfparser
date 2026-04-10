from __future__ import annotations

from typing import Protocol

from ..models.elements import Figure, Formula, Table, TextChunk, VisualPage


class IndexWriter(Protocol):
    """Writes extracted elements into the vector database.

    Each upsert method handles the element type's specific embedding format.
    Swapping this adapter changes the vector DB backend without touching pipelines.
    """

    def ensure_collection(
        self, collection: str, dim: int, is_multi_vector: bool = False
    ) -> None:
        """Create the collection if it does not already exist. Idempotent."""
        ...

    def upsert_visual(self, collection: str, pages: list[VisualPage]) -> None:
        """Write visual page embeddings. Handles multi-vector storage internally."""
        ...

    def upsert_text(self, collection: str, chunks: list[TextChunk]) -> None:
        """Write text chunk embeddings."""
        ...

    def upsert_tables(self, collection: str, tables: list[Table]) -> None:
        """Write table embeddings."""
        ...

    def upsert_formulas(self, collection: str, formulas: list[Formula]) -> None:
        """Write formula embeddings."""
        ...

    def upsert_figures(self, collection: str, figures: list[Figure]) -> None:
        """Write figure embeddings."""
        ...
