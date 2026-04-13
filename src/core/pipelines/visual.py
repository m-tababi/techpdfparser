from __future__ import annotations

import logging
from pathlib import Path

from ...utils.ids import generate_element_id
from ...utils.jsonl import write_jsonl
from ...utils.manifest import ManifestBuilder, record_tool_version
from ...utils.runtime import release_runtime_resources
from ...utils.storage import StorageManager
from ...utils.timing import timed
from ..config import VisualPipelineConfig
from ..indexing import (
    ResolvedIndexLayout,
    VectorSchema,
    get_visual_vector_schema,
    layout_metadata,
)
from ..interfaces.indexer import IndexWriter
from ..interfaces.renderer import PageRenderer
from ..interfaces.visual import VisualEmbedder
from ..models.document import DocumentMeta
from ..models.elements import VisualPage

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
        index_layout: ResolvedIndexLayout | None = None,
        fail_on_schema_mismatch: bool = True,
    ) -> None:
        self.renderer = renderer
        self.embedder = embedder
        self.index_writer = index_writer
        self.storage = storage
        self.config = config
        self.index_layout = index_layout
        self.fail_on_schema_mismatch = fail_on_schema_mismatch

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

        collection_name = self._collection_name()
        schema = self._vector_schema()
        self.index_writer.ensure_collection(
            collection_name,
            schema,
            fail_on_schema_mismatch=self.fail_on_schema_mismatch,
        )

        with timed("render_and_embed") as t:
            pages = self._render_and_embed(pdf_path, doc_meta, run_dir)
        logger.info(f"Rendered and embedded {len(pages)} pages in {t.elapsed_seconds:.2f}s")
        release_runtime_resources(self.embedder)

        with timed("index_write") as t:
            self.index_writer.upsert_visual(collection_name, pages)
        logger.info(f"Indexed {len(pages)} pages in {t.elapsed_seconds:.2f}s")

        self._write_outputs(run_dir, pages, manifest, collection_name, schema)
        self.storage.update_document_index(
            doc_meta.doc_id,
            str(pdf_path),
            run_id,
            "visual",
            extra=layout_metadata(self.index_layout) if self.index_layout else None,
        )

        return pages

    def _collection_name(self) -> str:
        if self.index_layout is not None:
            return self.index_layout.collections["visual"]
        return self.config.collection

    def _vector_schema(self):
        if self.index_layout is not None:
            return self.index_layout.vector_schemas["visual"]
        return get_visual_vector_schema(self.embedder)

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
        self,
        run_dir: Path,
        pages: list[VisualPage],
        manifest: ManifestBuilder,
        collection_name: str,
        schema: VectorSchema,
    ) -> None:
        write_jsonl(run_dir / "elements.jsonl", pages)
        record_tool_version(manifest, self.renderer)
        record_tool_version(manifest, self.embedder)
        manifest.set_counts(pages=len(pages), elements=len(pages))
        if self.index_layout is not None:
            manifest.set_index_info(
                backend=self.index_layout.backend,
                namespace=self.index_layout.namespace or "legacy",
                collections=[collection_name],
                upserted=len(pages),
                adapter_signatures=dict(self.index_layout.adapter_signatures),
                vector_schemas={"visual": schema.model_dump(mode="json")},
            )
        else:
            manifest.set_qdrant_info(collection_name, len(pages))
        manifest.write(run_dir)
