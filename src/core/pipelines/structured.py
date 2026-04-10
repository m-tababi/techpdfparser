from __future__ import annotations

import logging
from pathlib import Path

from ..config import StructuredPipelineConfig
from ..interfaces.embedder import TextEmbedder
from ..interfaces.figure import FigureDescriptor
from ..interfaces.formula import FormulaExtractor
from ..interfaces.indexer import IndexWriter
from ..interfaces.parser import StructuredParser
from ..models.document import DocumentMeta
from ..models.elements import Figure, Formula, Table
from ...utils.storage import StorageManager
from ...utils.timing import timed

logger = logging.getLogger("techpdfparser.pipelines.structured")


class StructuredPipeline:
    """Orchestrates: parse structured elements → enrich → embed → index.

    The parser locates tables, formulas, and figures. The formula_extractor
    and figure_descriptor are enrichment steps that run on the located regions.
    All three can be swapped independently via config.
    """

    def __init__(
        self,
        parser: StructuredParser,
        formula_extractor: FormulaExtractor,
        figure_descriptor: FigureDescriptor,
        embedder: TextEmbedder,
        index_writer: IndexWriter,
        storage: StorageManager,
        config: StructuredPipelineConfig,
    ) -> None:
        self.parser = parser
        self.formula_extractor = formula_extractor
        self.figure_descriptor = figure_descriptor
        self.embedder = embedder
        self.index_writer = index_writer
        self.storage = storage
        self.config = config

    def run(
        self, pdf_path: Path, doc_meta: DocumentMeta
    ) -> tuple[list[Table], list[Formula], list[Figure]]:
        """Run the full structured pipeline for one document."""
        logger.info(f"Structured pipeline start | doc={doc_meta.doc_id}")

        cols = self.config.collections
        for col in [cols.tables, cols.formulas, cols.figures]:
            self.index_writer.ensure_collection(col, self.embedder.embedding_dim)

        with timed("parse") as t:
            tables, formulas, figures = self.parser.parse(pdf_path, doc_meta.doc_id)
        logger.info(
            f"Parsed {len(tables)} tables, {len(formulas)} formulas, "
            f"{len(figures)} figures in {t.elapsed_seconds:.2f}s"
        )

        tables = self._embed_tables(tables)
        formulas = self._embed_formulas(formulas)
        figures = self._embed_figures(figures)

        self.index_writer.upsert_tables(cols.tables, tables)
        self.index_writer.upsert_formulas(cols.formulas, formulas)
        self.index_writer.upsert_figures(cols.figures, figures)

        return tables, formulas, figures

    def _embed_tables(self, tables: list[Table]) -> list[Table]:
        if not tables:
            return tables
        embeddings = self.embedder.embed([t.content for t in tables])
        for table, emb in zip(tables, embeddings):
            table.embedding = emb
        return tables

    def _embed_formulas(self, formulas: list[Formula]) -> list[Formula]:
        if not formulas:
            return formulas
        # Prefer LaTeX for embedding; fall back to plain-text content
        texts = [f.latex if f.latex else f.content for f in formulas]
        embeddings = self.embedder.embed(texts)
        for formula, emb in zip(formulas, embeddings):
            formula.embedding = emb
        return formulas

    def _embed_figures(self, figures: list[Figure]) -> list[Figure]:
        if not figures:
            return figures
        # Use VLM description if available, otherwise fall back to caption
        texts = [fig.description or fig.caption or "" for fig in figures]
        embeddings = self.embedder.embed(texts)
        for fig, emb in zip(figures, embeddings):
            fig.embedding = emb
        return figures
