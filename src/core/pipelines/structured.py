from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image as PILImage

from ..config import StructuredPipelineConfig
from ..interfaces.embedder import TextEmbedder
from ..interfaces.figure import FigureDescriptor
from ..interfaces.formula import FormulaExtractor
from ..interfaces.indexer import IndexWriter
from ..interfaces.parser import StructuredParser
from ..interfaces.renderer import PageRenderer
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
        renderer: PageRenderer | None = None,
    ) -> None:
        self.parser = parser
        self.formula_extractor = formula_extractor
        self.figure_descriptor = figure_descriptor
        self.embedder = embedder
        self.index_writer = index_writer
        self.storage = storage
        self.config = config
        # Optional: needed only for formula image cropping (PP-FormulaNet standalone)
        self.renderer = renderer

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

        formulas = self._enrich_formulas(formulas, pdf_path)
        figures = self._enrich_figures(figures)

        tables = self._embed_tables(tables)
        formulas = self._embed_formulas(formulas)
        figures = self._embed_figures(figures)

        self.index_writer.upsert_tables(cols.tables, tables)
        self.index_writer.upsert_formulas(cols.formulas, formulas)
        self.index_writer.upsert_figures(cols.figures, figures)

        return tables, formulas, figures

    def _enrich_figures(self, figures: list[Figure]) -> list[Figure]:
        """Run VLM description on figures that have an image but no description yet."""
        for fig in figures:
            if fig.description is None and fig.image_path:
                img = PILImage.open(fig.image_path)
                fig.description = self.figure_descriptor.describe(img)
        return figures

    def _enrich_formulas(self, formulas: list[Formula], pdf_path: Path) -> list[Formula]:
        """Crop formula regions and run PP-FormulaNet to get LaTeX for bbox-only formulas.

        Skipped entirely when no renderer is provided (e.g. MinerU already supplied LaTeX).
        """
        if self.renderer is None:
            return formulas
        for formula in formulas:
            if formula.latex or formula.bbox is None:
                continue
            page_img = self.renderer.render_page(pdf_path, formula.page_number)
            bb = formula.bbox
            crop = page_img.crop((int(bb.x0), int(bb.y0), int(bb.x1), int(bb.y1)))
            results = self.formula_extractor.extract(
                crop, doc_id=formula.doc_id, page_number=formula.page_number
            )
            if results:
                formula.latex = results[0].latex
                formula.content = results[0].content or results[0].latex
        return formulas

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
