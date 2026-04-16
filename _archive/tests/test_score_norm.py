"""Test MinMaxScoreNormFusion correctness."""

from src.adapters.fusion.score_norm import MinMaxScoreNormFusion
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


class TestMinMaxScoreNormFusion:
    def test_empty_input(self):
        fusion = MinMaxScoreNormFusion()
        assert fusion.fuse([]) == []

    def test_single_list_rank_order_preserved(self):
        fusion = MinMaxScoreNormFusion()
        result_list = [
            make_result("a", 0.9),
            make_result("b", 0.5),
            make_result("c", 0.1),
        ]
        fused = fusion.fuse([result_list])
        ids = [r.element.object_id for r in fused]
        assert ids[0] == "a"
        assert ids[1] == "b"
        assert ids[2] == "c"

    def test_equal_scores_produce_equal_fused_scores(self):
        fusion = MinMaxScoreNormFusion()
        # All scores identical → normalised to 1.0 → all fused scores equal
        result_list = [make_result("a", 0.7), make_result("b", 0.7)]
        fused = fusion.fuse([result_list])
        assert fused[0].fused_score == fused[1].fused_score

    def test_overlap_boosted_by_second_list(self):
        fusion = MinMaxScoreNormFusion()
        # "a" appears in both lists, "b" only in list1
        list1 = [make_result("a", 0.9), make_result("b", 0.8)]
        list2 = [make_result("a", 0.5, "visual_pages")]
        fused = fusion.fuse([list1, list2])
        assert fused[0].element.object_id == "a"

    def test_weighted_fusion_amplifies_high_weight_list(self):
        fusion = MinMaxScoreNormFusion()
        # "b" is top in list1 (weight=10), "a" is top in list2 (weight=1)
        list1 = [make_result("b", 0.9), make_result("a", 0.1)]
        list2 = [make_result("a", 0.9), make_result("b", 0.1)]
        fused = fusion.fuse([list1, list2], weights=[10.0, 1.0])
        assert fused[0].element.object_id == "b"

    def test_ranks_assigned_sequentially(self):
        fusion = MinMaxScoreNormFusion()
        result_list = [make_result("x", 0.8), make_result("y", 0.4)]
        fused = fusion.fuse([result_list])
        assert fused[0].rank == 1
        assert fused[1].rank == 2

    def test_source_scores_preserved(self):
        fusion = MinMaxScoreNormFusion()
        list1 = [make_result("a", 0.9, "text_chunks")]
        list2 = [make_result("a", 0.6, "visual_pages")]
        fused = fusion.fuse([list1, list2])
        assert "text_chunks" in fused[0].source_scores
        assert "visual_pages" in fused[0].source_scores
