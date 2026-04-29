from __future__ import annotations

from typing import Protocol

from ..indexing import VectorSchema
from ..models.elements import Figure, Formula, Table, TextChunk, VisualPage


class IndexWriter(Protocol):
    """Writes extracted elements into the vector database.

    Each upsert method handles the element type's specific embedding format.
    Swapping this adapter changes the vector DB backend without touching pipelines.
    """

    def ensure_collection(
        self,
        collection: str,
        schema: VectorSchema,
        fail_on_schema_mismatch: bool = True,
    ) -> None:
        """Create the collection or validate the existing schema."""
        ...

    def get_collection_schema(self, collection: str) -> VectorSchema | None:
        """Return the current schema or None when the collection does not exist."""
        ...

    def healthcheck(self) -> None:
        """Raise when the backend is unreachable or misconfigured."""
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
