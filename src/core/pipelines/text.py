from __future__ import annotations

import logging
from pathlib import Path

from ...utils.jsonl import write_jsonl
from ...utils.manifest import ManifestBuilder, record_tool_version
from ...utils.sections import write_sections
from ...utils.storage import StorageManager
from ...utils.timing import timed
from ..config import TextPipelineConfig
from ..indexing import (
    ResolvedIndexLayout,
    VectorSchema,
    get_text_vector_schema,
    layout_metadata,
)
from ..interfaces.chunker import TextChunker
from ..interfaces.embedder import TextEmbedder
from ..interfaces.extractor import TextExtractor
from ..interfaces.indexer import IndexWriter
from ..models.document import DocumentMeta
from ..models.elements import TextChunk

logger = logging.getLogger("techpdfparser.pipelines.text")


class TextPipeline:
    """Orchestrates: extract text → chunk → embed → index.

    Swapping olmOCR2 for another extractor or BGE-M3 for another embedder
    only requires passing a different adapter — the pipeline is unchanged.
    """

    def __init__(
        self,
        extractor: TextExtractor,
        chunker: TextChunker,
        embedder: TextEmbedder,
        index_writer: IndexWriter,
        storage: StorageManager,
        config: TextPipelineConfig,
        index_layout: ResolvedIndexLayout | None = None,
        fail_on_schema_mismatch: bool = True,
    ) -> None:
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.index_writer = index_writer
        self.storage = storage
        self.config = config
        self.index_layout = index_layout
        self.fail_on_schema_mismatch = fail_on_schema_mismatch

    def run(self, pdf_path: Path, doc_meta: DocumentMeta) -> list[TextChunk]:
        """Run the full text pipeline for one document."""
        tool_suffix = f"{self.extractor.tool_name}_{self.embedder.tool_name}"
        run_dir = self.storage.run_dir(doc_meta.doc_id, "text", tool_suffix)
        run_id = self.storage.run_id_from_dir(run_dir)
        manifest = ManifestBuilder(
            run_id=run_id,
            pipeline="text",
            doc_id=doc_meta.doc_id,
            source_file=str(pdf_path),
            tools={
                "extractor": self.config.extractor,
                "chunker": self.config.chunker,
                "embedder": self.config.embedder,
            },
        )
        logger.info(f"Text pipeline start | doc={doc_meta.doc_id} | run={run_id}")

        collection_name = self._collection_name()
        schema = self._vector_schema()
        self.index_writer.ensure_collection(
            collection_name,
            schema,
            fail_on_schema_mismatch=self.fail_on_schema_mismatch,
        )

        with timed("extract") as t:
            raw_blocks = self.extractor.extract_all(pdf_path, doc_meta.doc_id)
        logger.info(f"Extracted {len(raw_blocks)} blocks in {t.elapsed_seconds:.2f}s")

        # Persist raw extractor output before chunking
        write_jsonl(run_dir / "raw_blocks.jsonl", raw_blocks)

        # Write section markers when the extractor supports it (pymupdf_structured)
        if hasattr(self.extractor, "get_markers"):
            markers = self.extractor.get_markers(pdf_path)
            if markers:
                write_sections(run_dir / "sections.json", markers)
                logger.info(f"Wrote {len(markers)} section markers")

        chunks = self.chunker.chunk(raw_blocks)
        logger.info(f"Chunked into {len(chunks)} chunks")

        with timed("embed") as t:
            chunks = self._embed(chunks)
        logger.info(f"Embedded {len(chunks)} chunks in {t.elapsed_seconds:.2f}s")

        with timed("index_write") as t:
            self.index_writer.upsert_text(collection_name, chunks)
        logger.info(f"Indexed {len(chunks)} chunks in {t.elapsed_seconds:.2f}s")

        self._write_outputs(run_dir, raw_blocks, chunks, manifest, collection_name, schema)
        self.storage.update_document_index(
            doc_meta.doc_id,
            str(pdf_path),
            run_id,
            "text",
            extra=layout_metadata(self.index_layout) if self.index_layout else None,
        )

        return chunks

    def _collection_name(self) -> str:
        if self.index_layout is not None:
            return self.index_layout.collections["text"]
        return self.config.collection

    def _vector_schema(self):
        if self.index_layout is not None:
            return self.index_layout.vector_schemas["text"]
        return get_text_vector_schema(self.embedder)

    def _embed(self, chunks: list[TextChunk]) -> list[TextChunk]:
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)
        if len(embeddings) != len(chunks):
            raise ValueError(
                "Text embedder returned "
                f"{len(embeddings)} vectors for {len(chunks)} chunks"
            )
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks

    def _write_outputs(
        self,
        run_dir: Path,
        raw_blocks: list[TextChunk],
        chunks: list[TextChunk],
        manifest: ManifestBuilder,
        collection_name: str,
        schema: VectorSchema,
    ) -> None:
        # raw_blocks.jsonl already written before chunking; write chunks now
        write_jsonl(run_dir / "chunks.jsonl", chunks)
        record_tool_version(manifest, self.extractor)
        record_tool_version(manifest, self.embedder)
        manifest.set_counts(raw_blocks=len(raw_blocks), chunks=len(chunks))
        if self.index_layout is not None:
            manifest.set_index_info(
                backend=self.index_layout.backend,
                namespace=self.index_layout.namespace or "legacy",
                collections=[collection_name],
                upserted=len(chunks),
                adapter_signatures=dict(self.index_layout.adapter_signatures),
                vector_schemas={"text": schema.model_dump(mode="json")},
            )
        else:
            manifest.set_qdrant_info(collection_name, len(chunks))
        manifest.write(run_dir)
