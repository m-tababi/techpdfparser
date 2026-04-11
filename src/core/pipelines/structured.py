from __future__ import annotations

import logging
import shutil
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
from ...utils.jsonl import write_jsonl
from ...utils.manifest import ManifestBuilder
from ...utils.sections import SectionMarker, assign_sections, load_sections
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
        tool_suffix = (
            f"{self.config.parser}_{self.config.formula_extractor}"
            f"_{self.config.figure_descriptor}"
        )
        run_dir = self.storage.run_dir(doc_meta.doc_id, "structured", tool_suffix)
        run_id = self.storage.run_id_from_dir(run_dir)
        manifest = ManifestBuilder(
            run_id=run_id,
            pipeline="structured",
            doc_id=doc_meta.doc_id,
            source_file=str(pdf_path),
            tools={
                "parser": self.config.parser,
                "formula_extractor": self.config.formula_extractor,
                "figure_descriptor": self.config.figure_descriptor,
            },
        )
        logger.info(f"Structured pipeline start | doc={doc_meta.doc_id} | run={run_id}")

        cols = self.config.collections
        for col in [cols.tables, cols.formulas, cols.figures]:
            self.index_writer.ensure_collection(col, self.embedder.embedding_dim)

        with timed("parse") as t:
            tables, formulas, figures = self.parser.parse(pdf_path, doc_meta.doc_id)
        logger.info(
            f"Parsed {len(tables)} tables, {len(formulas)} formulas, "
            f"{len(figures)} figures in {t.elapsed_seconds:.2f}s"
        )

        # Move figure images from parser's tempdir into the run directory
        figures = self._persist_figures(figures, run_dir)

        # Attach section context from the most recent text run (if available)
        sections_path = self.storage.latest_text_sections(doc_meta.doc_id)
        section_source: str | None = None
        if sections_path is not None:
            markers = load_sections(sections_path)
            all_elements: list = [*tables, *formulas, *figures]
            assign_sections(all_elements, markers)
            # Record which text run provided section data for traceability
            section_source = sections_path.parent.name
            logger.info(f"Linked sections from {section_source}")

        formulas = self._enrich_formulas(formulas, pdf_path)
        figures = self._enrich_figures(figures)

        tables = self._embed_tables(tables)
        formulas = self._embed_formulas(formulas)
        figures = self._embed_figures(figures)

        self.index_writer.upsert_tables(cols.tables, tables)
        self.index_writer.upsert_formulas(cols.formulas, formulas)
        self.index_writer.upsert_figures(cols.figures, figures)

        self._write_outputs(run_dir, tables, formulas, figures, manifest, section_source)
        self.storage.update_document_index(
            doc_meta.doc_id, str(pdf_path), run_id, "structured"
        )

        return tables, formulas, figures

    def _persist_figures(self, figures: list[Figure], run_dir: Path) -> list[Figure]:
        """Move figure images from the parser's tempdir into the run directory.

        The parser writes to a tempdir it owns; the pipeline is responsible for
        persisting images to a stable location before the tempdir is cleaned up.
        """
        for fig in figures:
            if not fig.image_path:
                continue
            src = Path(fig.image_path)
            if not src.exists():
                continue
            dst = self.storage.figure_path(run_dir, fig.page_number, 0)
            # Use a unique name based on the original filename to avoid collisions
            dst = dst.parent / src.name
            shutil.move(str(src), str(dst))
            fig.image_path = str(dst)
            fig.raw_output_path = str(dst)
        return figures

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

    def _write_outputs(
        self,
        run_dir: Path,
        tables: list[Table],
        formulas: list[Formula],
        figures: list[Figure],
        manifest: ManifestBuilder,
        section_source: str | None = None,
    ) -> None:
        write_jsonl(run_dir / "tables.jsonl", tables)
        write_jsonl(run_dir / "formulas.jsonl", formulas)
        write_jsonl(run_dir / "figures.jsonl", figures)
        manifest.set_counts(
            tables=len(tables), formulas=len(formulas), figures=len(figures)
        )
        cols = self.config.collections
        manifest.set_qdrant_info(
            f"{cols.tables},{cols.formulas},{cols.figures}",
            len(tables) + len(formulas) + len(figures),
        )
        if section_source is not None:
            manifest.config["section_source"] = section_source
        manifest.write(run_dir)
