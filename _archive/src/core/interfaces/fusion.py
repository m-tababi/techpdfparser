from __future__ import annotations

from typing import Protocol

from ..models.results import FusionResult, RetrievalResult


class FusionEngine(Protocol):
    """Merges and reranks result lists from multiple retrieval pipelines.

    Implementations can use RRF, score normalization, learned rerankers, etc.
    Swap adapters to compare fusion strategies without changing retrieval code.
    """

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        weights: list[float] | None = None,
    ) -> list[FusionResult]:
        """Fuse multiple ranked result lists into a single ranked list.

        `weights` controls the relative importance of each input list.
        Defaults to equal weights when None.
        """
        ...
