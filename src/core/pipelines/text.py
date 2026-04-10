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
        logger.info(f"Text pipeline start | doc={doc_meta.doc_id}")

        self.index_writer.ensure_collection(
            self.config.collection, self.embedder.embedding_dim
        )

        with timed("extract") as t:
            raw_blocks = self.extractor.extract_all(pdf_path, doc_meta.doc_id)
        logger.info(f"Extracted {len(raw_blocks)} blocks in {t.elapsed_seconds:.2f}s")

        chunks = self.chunker.chunk(raw_blocks)
        logger.info(f"Chunked into {len(chunks)} chunks")

        with timed("embed") as t:
            chunks = self._embed(chunks)
        logger.info(f"Embedded {len(chunks)} chunks in {t.elapsed_seconds:.2f}s")

        with timed("index_write") as t:
            self.index_writer.upsert_text(self.config.collection, chunks)
        logger.info(f"Indexed {len(chunks)} chunks in {t.elapsed_seconds:.2f}s")

        return chunks

    def _embed(self, chunks: list[TextChunk]) -> list[TextChunk]:
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks
