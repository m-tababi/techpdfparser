from __future__ import annotations

from ...core.models.results import FusionResult, RetrievalResult
from ...core.registry import register_fusion_engine

# Standard RRF constant from Cormack et al. (2009).
# k=60 softens rank differences; lower values amplify top-rank advantage.
_RRF_K = 60


@register_fusion_engine("rrf")
class ReciprocalRankFusion:
    """Fuses result lists using Reciprocal Rank Fusion (RRF).

    RRF is score-scale-agnostic, making it ideal for fusing results from
    visual, text, and structured-element retrievers that operate in
    completely different score spaces.

    Replace with a learned reranker or score-normalization fusion by
    registering under a different name.

    Reference: Cormack, Clarke & Buettcher, SIGIR 2009.
    """

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        weights: list[float] | None = None,
    ) -> list[FusionResult]:
        """Fuse multiple ranked lists into a single ranked list via RRF."""
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        scores: dict[str, float] = {}
        source_scores: dict[str, dict[str, float]] = {}
        elements: dict[str, RetrievalResult] = {}

        for result_list, weight in zip(result_lists, weights):
            for rank, result in enumerate(result_list, start=1):
                key = result.element.object_id
                rrf_score = weight / (_RRF_K + rank)
                scores[key] = scores.get(key, 0.0) + rrf_score
                source_scores.setdefault(key, {})[result.collection] = result.score
                # Keep the first-seen result for each element (highest rank wins)
                elements.setdefault(key, result)

        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

        return [
            FusionResult(
                element=elements[key].element,
                fused_score=scores[key],
                source_scores=source_scores[key],
                rank=i + 1,
            )
            for i, key in enumerate(ranked_keys)
        ]
