from __future__ import annotations

import logging
from pathlib import Path

from ..config import TextPipelineConfig
from ..interfaces.chunker import TextChunker
from ..interfaces.embedder import TextEmbedder
from ..interfaces.extractor import TextExtractor
from ..interfaces.indexer import IndexWriter
from ..models.document import DocumentMeta
from ..models.elements import TextChunk
from ...utils.jsonl import write_jsonl
from ...utils.manifest import ManifestBuilder
from ...utils.sections import write_sections
from ...utils.storage import StorageManager
from ...utils.timing import timed

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
    ) -> None:
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.index_writer = index_writer
        self.storage = storage
        self.config = config

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

        self.index_writer.ensure_collection(
            self.config.collection, self.embedder.embedding_dim
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
            self.index_writer.upsert_text(self.config.collection, chunks)
        logger.info(f"Indexed {len(chunks)} chunks in {t.elapsed_seconds:.2f}s")

        self._write_outputs(run_dir, raw_blocks, chunks, manifest)
        self.storage.update_document_index(
            doc_meta.doc_id, str(pdf_path), run_id, "text"
        )

        return chunks

    def _embed(self, chunks: list[TextChunk]) -> list[TextChunk]:
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks

    def _write_outputs(
        self,
        run_dir: Path,
        raw_blocks: list[TextChunk],
        chunks: list[TextChunk],
        manifest: ManifestBuilder,
    ) -> None:
        # raw_blocks.jsonl already written before chunking; write chunks now
        write_jsonl(run_dir / "chunks.jsonl", chunks)
        manifest.set_tool_version(self.extractor.tool_name, self.extractor.tool_version)
        manifest.set_counts(raw_blocks=len(raw_blocks), chunks=len(chunks))
        manifest.set_qdrant_info(self.config.collection, len(chunks))
        manifest.write(run_dir)
