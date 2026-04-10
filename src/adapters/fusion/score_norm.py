from __future__ import annotations

from ...core.models.results import FusionResult, RetrievalResult
from ...core.registry import register_fusion_engine


@register_fusion_engine("score_norm")
class MinMaxScoreNormFusion:
    """Fuses result lists via min-max normalisation then weighted average.

    Unlike RRF, this preserves relative score magnitude within each list —
    useful when retriever scores are calibrated and carry semantic meaning.
    Swap in via config `fusion_engine: score_norm`.
    """

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        weights: list[float] | None = None,
    ) -> list[FusionResult]:
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        scores: dict[str, float] = {}
        source_scores: dict[str, dict[str, float]] = {}
        elements: dict[str, RetrievalResult] = {}

        for result_list, weight in zip(result_lists, weights):
            normalised = _normalise(result_list)
            for result, norm_score in zip(result_list, normalised):
                key = result.element.object_id
                scores[key] = scores.get(key, 0.0) + weight * norm_score
                source_scores.setdefault(key, {})[result.collection] = result.score
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


def _normalise(results: list[RetrievalResult]) -> list[float]:
    """Min-max normalise scores to [0, 1]. Returns 1.0 for all when range is 0."""
    if not results:
        return []
    raw = [r.score for r in results]
    lo, hi = min(raw), max(raw)
    if hi == lo:
        return [1.0] * len(raw)
    return [(s - lo) / (hi - lo) for s in raw]
