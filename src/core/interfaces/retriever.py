from __future__ import annotations

from typing import Protocol

from ..models.results import RetrievalResult


class RetrievalEngine(Protocol):
    """Queries vector collections to retrieve relevant elements.

    Both visual (multi-vector MaxSim) and text (cosine) search are supported.
    A single engine can serve all collections in one vector DB instance.
    """

    def search_visual(
        self,
        collection: str,
        query_embedding: list[list[float]],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search visual pages using a multi-vector query (MaxSim scoring)."""
        ...

    def search_text(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search text, tables, formulas, or figures using a dense query vector."""
        ...
