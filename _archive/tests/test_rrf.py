"""Test Reciprocal Rank Fusion correctness."""

import pytest

from src.adapters.fusion.rrf import ReciprocalRankFusion
from src.core.models.elements import TextChunk
from src.core.models.results import RetrievalResult


def make_result(object_id: str, score: float, collection: str = "text_chunks") -> RetrievalResult:
    chunk = TextChunk(
        object_id=object_id,
        doc_id="doc1",
        source_file="test.pdf",
        page_number=0,
        tool_name="test",
        tool_version="1.0",
        content=f"Content for {object_id}",
    )
    return RetrievalResult(element=chunk, score=score, collection=collection)


class TestReciprocalRankFusion:
    def test_empty_input(self):
        rrf = ReciprocalRankFusion()
        results = rrf.fuse([])
        assert results == []

    def test_single_list_preserves_rank_order(self):
        rrf = ReciprocalRankFusion()
        result_list = [
            make_result("a", 0.9),
            make_result("b", 0.7),
            make_result("c", 0.5),
        ]
        fused = rrf.fuse([result_list])
        ids = [r.element.object_id for r in fused]
        # First ranked element gets highest RRF score → should stay first
        assert ids[0] == "a"
        assert ids[1] == "b"
        assert ids[2] == "c"

    def test_ranks_assigned(self):
        rrf = ReciprocalRankFusion()
        result_list = [make_result("a", 0.9), make_result("b", 0.5)]
        fused = rrf.fuse([result_list])
        assert fused[0].rank == 1
        assert fused[1].rank == 2

    def test_fusion_boosts_overlap(self):
        rrf = ReciprocalRankFusion()
        # "a" appears in both lists at rank 1 → should beat "b" and "c"
        list1 = [make_result("a", 0.9), make_result("b", 0.8)]
        list2 = [make_result("a", 0.7, "visual_pages"), make_result("c", 0.6, "visual_pages")]
        fused = rrf.fuse([list1, list2])
        assert fused[0].element.object_id == "a"

    def test_source_scores_populated(self):
        rrf = ReciprocalRankFusion()
        list1 = [make_result("a", 0.9, "text_chunks")]
        list2 = [make_result("a", 0.8, "visual_pages")]
        fused = rrf.fuse([list1, list2])
        assert "text_chunks" in fused[0].source_scores
        assert "visual_pages" in fused[0].source_scores

    def test_weights_affect_order(self):
        rrf = ReciprocalRankFusion()
        # "b" is ranked first in list1 (weight=10), "a" first in list2 (weight=1)
        # With high weight on list1, "b" should beat "a"
        list1 = [make_result("b", 0.9), make_result("a", 0.5)]
        list2 = [make_result("a", 0.9), make_result("b", 0.5)]
        fused_equal = rrf.fuse([list1, list2], weights=[1.0, 1.0])
        fused_weighted = rrf.fuse([list1, list2], weights=[10.0, 1.0])
        # With equal weights, tie may go either way — just check no crash
        assert len(fused_equal) == 2
        # With heavy weight on list1, "b" should win
        assert fused_weighted[0].element.object_id == "b"

    def test_fused_score_is_positive(self):
        rrf = ReciprocalRankFusion()
        result_list = [make_result("a", 0.9)]
        fused = rrf.fuse([result_list])
        assert fused[0].fused_score > 0
