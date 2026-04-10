"""Test UnifiedRetriever orchestration with mock adapters."""

from unittest.mock import MagicMock

from src.core.retrieval import UnifiedRetriever


def make_retriever(**kwargs) -> UnifiedRetriever:
    defaults = dict(
        retrieval_engine=MagicMock(),
        visual_embedder=MagicMock(),
        text_embedder=MagicMock(),
        fusion_engine=MagicMock(),
        visual_collection="visual_pages",
        text_collection="text_chunks",
        tables_collection="tables",
        formulas_collection="formulas",
        figures_collection="figures",
    )
    defaults.update(kwargs)
    return UnifiedRetriever(**defaults)


class TestUnifiedRetriever:
    def test_query_calls_both_embedders(self):
        retriever = make_retriever()
        retriever.visual_embedder.embed_query.return_value = [[0.1, 0.2]]
        retriever.text_embedder.embed_query.return_value = [0.3, 0.4]
        retriever.retrieval_engine.search_visual.return_value = []
        retriever.retrieval_engine.search_text.return_value = []
        retriever.fusion_engine.fuse.return_value = []

        retriever.query("test query")

        retriever.visual_embedder.embed_query.assert_called_once_with("test query")
        retriever.text_embedder.embed_query.assert_called_once_with("test query")

    def test_query_searches_all_five_collections(self):
        retriever = make_retriever()
        retriever.visual_embedder.embed_query.return_value = [[0.1]]
        retriever.text_embedder.embed_query.return_value = [0.2]
        retriever.retrieval_engine.search_visual.return_value = []
        retriever.retrieval_engine.search_text.return_value = []
        retriever.fusion_engine.fuse.return_value = []

        retriever.query("test query")

        retriever.retrieval_engine.search_visual.assert_called_once_with(
            "visual_pages", [[0.1]], 10
        )
        assert retriever.retrieval_engine.search_text.call_count == 4

    def test_fusion_called_with_five_result_lists(self):
        retriever = make_retriever()
        retriever.visual_embedder.embed_query.return_value = [[0.1]]
        retriever.text_embedder.embed_query.return_value = [0.2]
        retriever.retrieval_engine.search_visual.return_value = ["v"]
        retriever.retrieval_engine.search_text.return_value = ["t"]
        retriever.fusion_engine.fuse.return_value = []

        retriever.query("test query")

        call_args = retriever.fusion_engine.fuse.call_args
        result_lists = call_args[0][0]
        assert len(result_lists) == 5

    def test_query_passes_weights_to_fusion(self):
        retriever = make_retriever()
        retriever.visual_embedder.embed_query.return_value = [[0.1]]
        retriever.text_embedder.embed_query.return_value = [0.2]
        retriever.retrieval_engine.search_visual.return_value = []
        retriever.retrieval_engine.search_text.return_value = []
        retriever.fusion_engine.fuse.return_value = []

        weights = [2.0, 1.0, 1.0, 1.0, 1.0]
        retriever.query("test query", weights=weights)

        call_args = retriever.fusion_engine.fuse.call_args
        assert call_args[0][1] == weights

    def test_query_returns_fusion_output(self):
        retriever = make_retriever()
        retriever.visual_embedder.embed_query.return_value = [[0.1]]
        retriever.text_embedder.embed_query.return_value = [0.2]
        retriever.retrieval_engine.search_visual.return_value = []
        retriever.retrieval_engine.search_text.return_value = []
        sentinel = object()
        retriever.fusion_engine.fuse.return_value = sentinel

        result = retriever.query("test query")
        assert result is sentinel
