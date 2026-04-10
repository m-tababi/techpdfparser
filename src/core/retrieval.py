from __future__ import annotations

from .interfaces.embedder import TextEmbedder
from .interfaces.fusion import FusionEngine
from .interfaces.retriever import RetrievalEngine
from .interfaces.visual import VisualEmbedder
from .models.results import FusionResult


class UnifiedRetriever:
    """Queries all five collections and fuses the results.

    One query produces ranked results across visual pages, text chunks,
    tables, formulas, and figures — regardless of which pipeline indexed them.
    """

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        visual_embedder: VisualEmbedder,
        text_embedder: TextEmbedder,
        fusion_engine: FusionEngine,
        visual_collection: str,
        text_collection: str,
        tables_collection: str,
        formulas_collection: str,
        figures_collection: str,
    ) -> None:
        self.retrieval_engine = retrieval_engine
        self.visual_embedder = visual_embedder
        self.text_embedder = text_embedder
        self.fusion_engine = fusion_engine
        self.visual_collection = visual_collection
        self.text_collection = text_collection
        self.tables_collection = tables_collection
        self.formulas_collection = formulas_collection
        self.figures_collection = figures_collection

    def query(
        self,
        query: str,
        top_k: int = 10,
        weights: list[float] | None = None,
    ) -> list[FusionResult]:
        """Query all collections and return fused, ranked results.

        `weights` controls per-collection importance in order:
        [visual, text, tables, formulas, figures]. Defaults to equal weights.
        """
        visual_emb = self.visual_embedder.embed_query(query)
        text_emb = self.text_embedder.embed_query(query)

        result_lists = [
            self.retrieval_engine.search_visual(self.visual_collection, visual_emb, top_k),
            self.retrieval_engine.search_text(self.text_collection, text_emb, top_k),
            self.retrieval_engine.search_text(self.tables_collection, text_emb, top_k),
            self.retrieval_engine.search_text(self.formulas_collection, text_emb, top_k),
            self.retrieval_engine.search_text(self.figures_collection, text_emb, top_k),
        ]

        return self.fusion_engine.fuse(result_lists, weights)
