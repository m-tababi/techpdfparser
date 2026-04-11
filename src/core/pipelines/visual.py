from __future__ import annotations

import logging
from pathlib import Path

from ..config import VisualPipelineConfig
from ..interfaces.indexer import IndexWriter
from ..interfaces.renderer import PageRenderer
from ..interfaces.visual import VisualEmbedder
from ..models.document import DocumentMeta
from ..models.elements import VisualPage
from ...utils.ids import generate_element_id
from ...utils.jsonl import write_jsonl
from ...utils.manifest import ManifestBuilder
from ...utils.storage import StorageManager
from ...utils.timing import timed

logger = logging.getLogger("techpdfparser.pipelines.visual")


class VisualPipeline:
    """Orchestrates: render pages → generate visual embeddings → index.

    All heavy work is delegated to injected adapters. Swapping ColQwen2.5
    for ColPali means passing a different `embedder` at construction time —
    the pipeline code stays unchanged.
    """

    def __init__(
        self,
        renderer: PageRenderer,
        embedder: VisualEmbedder,
        index_writer: IndexWriter,
        storage: StorageManager,
        config: VisualPipelineConfig,
    ) -> None:
        self.renderer = renderer
        self.embedder = embedder
        self.index_writer = index_writer
        self.storage = storage
        self.config = config

    def run(self, pdf_path: Path, doc_meta: DocumentMeta) -> list[VisualPage]:
        """Run the full visual pipeline for one document."""
        run_dir = self.storage.run_dir(
            doc_meta.doc_id, "visual", self.embedder.tool_name
        )
        run_id = self.storage.run_id_from_dir(run_dir)
        manifest = ManifestBuilder(
            run_id=run_id,
            pipeline="visual",
            doc_id=doc_meta.doc_id,
            source_file=str(pdf_path),
            tools={"renderer": self.config.renderer, "embedder": self.config.embedder},
        )
        logger.info(f"Visual pipeline start | doc={doc_meta.doc_id} | run={run_id}")

        self.index_writer.ensure_collection(
            self.config.collection,
            self.embedder.embedding_dim,
            is_multi_vector=self.embedder.is_multi_vector,
        )

        with timed("render_and_embed") as t:
            pages = self._render_and_embed(pdf_path, doc_meta, run_dir)
        logger.info(f"Rendered and embedded {len(pages)} pages in {t.elapsed_seconds:.2f}s")

        with timed("index_write") as t:
            self.index_writer.upsert_visual(self.config.collection, pages)
        logger.info(f"Indexed {len(pages)} pages in {t.elapsed_seconds:.2f}s")

        self._write_outputs(run_dir, pages, manifest)
        self.storage.update_document_index(
            doc_meta.doc_id, str(pdf_path), run_id, "visual"
        )

        return pages

    def _render_and_embed(
        self, pdf_path: Path, doc_meta: DocumentMeta, run_dir: Path
    ) -> list[VisualPage]:
        images = self.renderer.render_all(pdf_path)
        pages: list[VisualPage] = []

        for page_num, image in enumerate(images):
            image_path = self.storage.image_path(run_dir, page_num)
            image.save(str(image_path))
            embedding = self.embedder.embed_page(image)

            pages.append(
                VisualPage(
                    object_id=generate_element_id(
                        doc_meta.doc_id, page_num, "visual_page", self.embedder.tool_name
                    ),
                    doc_id=doc_meta.doc_id,
                    source_file=str(pdf_path),
                    page_number=page_num,
                    tool_name=self.embedder.tool_name,
                    tool_version=self.embedder.tool_version,
                    image_path=str(image_path),
                    raw_output_path=str(image_path),
                    embedding=embedding,
                )
            )

        return pages

    def _write_outputs(
        self, run_dir: Path, pages: list[VisualPage], manifest: ManifestBuilder
    ) -> None:
        write_jsonl(run_dir / "elements.jsonl", pages)
        manifest.set_tool_version(self.embedder.tool_name, self.embedder.tool_version)
        manifest.set_counts(pages=len(pages), elements=len(pages))
        manifest.set_qdrant_info(self.config.collection, len(pages))
        manifest.write(run_dir)
