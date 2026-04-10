"""Test pipeline orchestration using lightweight mock adapters.

No GPU, no models — just verifying that pipelines call their adapters
in the right order and produce correctly shaped output.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import (
    StructuredCollectionsConfig,
    StructuredPipelineConfig,
    TextPipelineConfig,
    VisualPipelineConfig,
)
from src.core.models.document import DocumentMeta
from src.core.models.elements import Figure, Formula, Table, TextChunk, VisualPage
from src.core.pipelines.structured import StructuredPipeline
from src.core.pipelines.text import TextPipeline
from src.core.pipelines.visual import VisualPipeline
from src.utils.ids import generate_doc_id
from src.utils.storage import StorageManager


def make_doc_meta(tmp_path: Path) -> DocumentMeta:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"fake")
    return DocumentMeta(
        doc_id=generate_doc_id(str(pdf)),
        source_file=str(pdf),
        total_pages=2,
        file_size_bytes=4,
    )


def make_storage(tmp_path: Path) -> StorageManager:
    return StorageManager(tmp_path / "outputs")


def make_visual_page(doc_id: str, page: int) -> VisualPage:
    return VisualPage(
        object_id=f"vp_{page}",
        doc_id=doc_id,
        source_file="test.pdf",
        page_number=page,
        tool_name="mock_embedder",
        tool_version="0.0",
        image_path=f"/out/{page}.png",
        embedding=[[0.1, 0.2]],
    )


def make_text_chunk(doc_id: str, page: int) -> TextChunk:
    return TextChunk(
        object_id=f"tc_{page}",
        doc_id=doc_id,
        source_file="test.pdf",
        page_number=page,
        tool_name="mock_extractor",
        tool_version="0.0",
        content=f"Text on page {page}",
    )


class TestVisualPipeline:
    def test_run_calls_adapters_and_returns_pages(self, tmp_path):
        doc_meta = make_doc_meta(tmp_path)
        storage = make_storage(tmp_path)

        pages = [make_visual_page(doc_meta.doc_id, i) for i in range(2)]

        renderer = MagicMock()
        renderer.render_all.return_value = [MagicMock(), MagicMock()]  # 2 PIL images

        embedder = MagicMock()
        embedder.tool_name = "mock_embedder"
        embedder.tool_version = "0.0"
        embedder.embedding_dim = 2
        embedder.is_multi_vector = True
        embedder.embed_page.return_value = [[0.1, 0.2], [0.3, 0.4]]

        index_writer = MagicMock()

        config = VisualPipelineConfig(renderer="mock", embedder="mock_embedder")
        pipeline = VisualPipeline(renderer, embedder, index_writer, storage, config)

        result = pipeline.run(Path(doc_meta.source_file), doc_meta)

        assert len(result) == 2
        assert all(isinstance(p, VisualPage) for p in result)
        renderer.render_all.assert_called_once()
        assert embedder.embed_page.call_count == 2
        index_writer.ensure_collection.assert_called_once()
        index_writer.upsert_visual.assert_called_once()


class TestTextPipeline:
    def test_run_calls_adapters_in_order(self, tmp_path):
        doc_meta = make_doc_meta(tmp_path)
        storage = make_storage(tmp_path)

        raw_blocks = [make_text_chunk(doc_meta.doc_id, i) for i in range(2)]
        chunked = raw_blocks  # chunker returns same blocks in this mock

        extractor = MagicMock()
        extractor.extract_all.return_value = raw_blocks

        chunker = MagicMock()
        chunker.chunk.return_value = chunked

        embedder = MagicMock()
        embedder.tool_name = "mock_embedder"
        embedder.embedding_dim = 4
        embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]] * len(chunked)

        index_writer = MagicMock()
        config = TextPipelineConfig()
        pipeline = TextPipeline(extractor, chunker, embedder, index_writer, storage, config)

        result = pipeline.run(Path(doc_meta.source_file), doc_meta)

        extractor.extract_all.assert_called_once()
        chunker.chunk.assert_called_once_with(raw_blocks)
        embedder.embed.assert_called_once()
        index_writer.upsert_text.assert_called_once()
        assert all(c.embedding is not None for c in result)


class TestStructuredPipeline:
    def test_run_embeds_and_indexes_all_types(self, tmp_path):
        doc_meta = make_doc_meta(tmp_path)
        storage = make_storage(tmp_path)

        table = Table(
            object_id="t0", doc_id=doc_meta.doc_id, source_file="test.pdf",
            page_number=0, tool_name="mock", tool_version="0.0",
            content="| a | b |",
        )
        formula = Formula(
            object_id="f0", doc_id=doc_meta.doc_id, source_file="test.pdf",
            page_number=0, tool_name="mock", tool_version="0.0",
            latex=r"x^2", content="x^2",
        )
        figure = Figure(
            object_id="fig0", doc_id=doc_meta.doc_id, source_file="test.pdf",
            page_number=0, tool_name="mock", tool_version="0.0",
            image_path="/out/fig.png", caption="A chart",
            # description pre-set so _enrich_figures skips file I/O in this test
            description="A chart",
        )

        parser = MagicMock()
        parser.parse.return_value = ([table], [formula], [figure])

        formula_extractor = MagicMock()
        figure_descriptor = MagicMock()

        embedder = MagicMock()
        embedder.embedding_dim = 4
        embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

        index_writer = MagicMock()
        config = StructuredPipelineConfig(
            collections=StructuredCollectionsConfig()
        )
        pipeline = StructuredPipeline(
            parser, formula_extractor, figure_descriptor, embedder, index_writer, storage, config
        )

        tables, formulas, figures = pipeline.run(Path(doc_meta.source_file), doc_meta)

        assert len(tables) == 1
        assert tables[0].embedding is not None
        assert len(formulas) == 1
        assert formulas[0].embedding is not None
        assert len(figures) == 1
        assert figures[0].embedding is not None
        index_writer.upsert_tables.assert_called_once()
        index_writer.upsert_formulas.assert_called_once()
        index_writer.upsert_figures.assert_called_once()

    def test_figure_description_enrichment(self, tmp_path):
        doc_meta = make_doc_meta(tmp_path)
        storage = make_storage(tmp_path)

        img_path = tmp_path / "fig.png"
        img_path.write_bytes(b"fake-png")

        figure = Figure(
            object_id="fig0", doc_id=doc_meta.doc_id, source_file="test.pdf",
            page_number=0, tool_name="mock", tool_version="0.0",
            image_path=str(img_path),
        )

        parser = MagicMock()
        parser.parse.return_value = ([], [], [figure])

        figure_descriptor = MagicMock()
        figure_descriptor.describe.return_value = "A bar chart"

        embedder = MagicMock()
        embedder.embedding_dim = 4
        embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

        config = StructuredPipelineConfig(collections=StructuredCollectionsConfig())
        pipeline = StructuredPipeline(
            parser, MagicMock(), figure_descriptor, embedder, MagicMock(), storage, config
        )

        with patch("src.core.pipelines.structured.PILImage.open", return_value=MagicMock()):
            _, _, figures = pipeline.run(Path(doc_meta.source_file), doc_meta)

        assert figures[0].description == "A bar chart"
        figure_descriptor.describe.assert_called_once()

    def test_formula_enrichment_with_renderer(self, tmp_path):
        doc_meta = make_doc_meta(tmp_path)
        storage = make_storage(tmp_path)

        from src.core.models.elements import BoundingBox
        formula = Formula(
            object_id="f0", doc_id=doc_meta.doc_id, source_file="test.pdf",
            page_number=0, tool_name="mock", tool_version="0.0",
            latex="", content="integral expression",
            bbox=BoundingBox(x0=10, y0=20, x1=50, y1=40),
        )

        parser = MagicMock()
        parser.parse.return_value = ([], [formula], [])

        enriched = Formula(
            object_id="f_enriched", doc_id=doc_meta.doc_id, source_file="test.pdf",
            page_number=0, tool_name="ppformulanet", tool_version="0.0",
            latex=r"\int_0^1 x\,dx", content=r"\int_0^1 x\,dx",
        )
        formula_extractor = MagicMock()
        formula_extractor.extract.return_value = [enriched]

        renderer = MagicMock()
        page_img = MagicMock()
        page_img.crop.return_value = MagicMock()
        renderer.render_page.return_value = page_img

        embedder = MagicMock()
        embedder.embedding_dim = 4
        embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

        config = StructuredPipelineConfig(collections=StructuredCollectionsConfig())
        pipeline = StructuredPipeline(
            parser, formula_extractor, MagicMock(), embedder, MagicMock(), storage, config,
            renderer=renderer,
        )

        _, formulas, _ = pipeline.run(Path(doc_meta.source_file), doc_meta)

        renderer.render_page.assert_called_once()
        formula_extractor.extract.assert_called_once()
        assert formulas[0].latex == r"\int_0^1 x\,dx"

    def test_formula_enrichment_skipped_without_renderer(self, tmp_path):
        doc_meta = make_doc_meta(tmp_path)
        storage = make_storage(tmp_path)

        from src.core.models.elements import BoundingBox
        formula = Formula(
            object_id="f0", doc_id=doc_meta.doc_id, source_file="test.pdf",
            page_number=0, tool_name="mock", tool_version="0.0",
            latex="", content="some formula",
            bbox=BoundingBox(x0=10, y0=20, x1=50, y1=40),
        )

        parser = MagicMock()
        parser.parse.return_value = ([], [formula], [])

        formula_extractor = MagicMock()

        embedder = MagicMock()
        embedder.embedding_dim = 4
        embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

        config = StructuredPipelineConfig(collections=StructuredCollectionsConfig())
        # renderer=None (default) → enrichment must be skipped
        pipeline = StructuredPipeline(
            parser, formula_extractor, MagicMock(), embedder, MagicMock(), storage, config
        )

        pipeline.run(Path(doc_meta.source_file), doc_meta)

        formula_extractor.extract.assert_not_called()
