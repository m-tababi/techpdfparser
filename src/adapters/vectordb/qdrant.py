from __future__ import annotations

from typing import Any

from ...core.models.elements import Figure, Formula, Table, TextChunk, VisualPage
from ...core.models.results import RetrievalResult
from ...core.registry import register_index_writer, register_retrieval_engine


def _connect(host: str, port: int):
    try:
        from qdrant_client import QdrantClient

        return QdrantClient(host=host, port=port)
    except ImportError:
        raise ImportError(
            "qdrant-client not installed. Run: pip install qdrant-client"
        )


def _base_payload(element: Any) -> dict:
    return {
        "doc_id": element.doc_id,
        "source_file": element.source_file,
        "page_number": element.page_number,
        "object_type": element.object_type,
        "tool_name": element.tool_name,
        "tool_version": element.tool_version,
        "bbox": element.bbox.model_dump() if element.bbox else None,
    }


@register_index_writer("qdrant")
class QdrantIndexWriter:
    """Writes extracted elements into Qdrant.

    Multi-vector pages (ColQwen2.5) use Qdrant's native multivector storage
    with MaxSim comparator. Single-vector elements use standard cosine similarity.

    Replace this class to switch to a different vector DB (Weaviate, Milvus, etc.)
    without touching any pipeline code.

    Requires: pip install qdrant-client
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_prefix: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._prefix = collection_prefix
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = _connect(self._host, self._port)
        return self._client

    def _col(self, name: str) -> str:
        return f"{self._prefix}{name}" if self._prefix else name

    def ensure_collection(
        self, collection: str, dim: int, is_multi_vector: bool = False
    ) -> None:
        client = self._get_client()
        from qdrant_client.models import Distance, MultiVectorComparator, MultiVectorConfig, VectorParams

        col = self._col(collection)
        existing = {c.name for c in client.get_collections().collections}
        if col in existing:
            return

        if is_multi_vector:
            # MaxSim over patch vectors — correct scoring for ColQwen/ColPali
            vector_config = VectorParams(
                size=dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ),
            )
        else:
            vector_config = VectorParams(size=dim, distance=Distance.COSINE)

        client.create_collection(collection_name=col, vectors_config=vector_config)

    def upsert_visual(self, collection: str, pages: list[VisualPage]) -> None:
        client = self._get_client()
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=page.object_id,
                vector=page.embedding,  # list[list[float]] for multi-vector
                payload={**_base_payload(page), "image_path": page.image_path},
            )
            for page in pages
            if page.embedding
        ]
        if points:
            client.upsert(collection_name=self._col(collection), points=points)

    def upsert_text(self, collection: str, chunks: list[TextChunk]) -> None:
        client = self._get_client()
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=chunk.object_id,
                vector=chunk.embedding,
                payload={**_base_payload(chunk), "content": chunk.content},
            )
            for chunk in chunks
            if chunk.embedding
        ]
        if points:
            client.upsert(collection_name=self._col(collection), points=points)

    def upsert_tables(self, collection: str, tables: list[Table]) -> None:
        client = self._get_client()
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=t.object_id,
                vector=t.embedding,
                payload={**_base_payload(t), "content": t.content},
            )
            for t in tables
            if t.embedding
        ]
        if points:
            client.upsert(collection_name=self._col(collection), points=points)

    def upsert_formulas(self, collection: str, formulas: list[Formula]) -> None:
        client = self._get_client()
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=f.object_id,
                vector=f.embedding,
                payload={**_base_payload(f), "latex": f.latex, "content": f.content},
            )
            for f in formulas
            if f.embedding
        ]
        if points:
            client.upsert(collection_name=self._col(collection), points=points)

    def upsert_figures(self, collection: str, figures: list[Figure]) -> None:
        client = self._get_client()
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=fig.object_id,
                vector=fig.embedding,
                payload={
                    **_base_payload(fig),
                    "image_path": fig.image_path,
                    "description": fig.description,
                    "caption": fig.caption,
                },
            )
            for fig in figures
            if fig.embedding
        ]
        if points:
            client.upsert(collection_name=self._col(collection), points=points)


@register_retrieval_engine("qdrant")
class QdrantRetrievalEngine:
    """Queries Qdrant collections to retrieve relevant elements.

    Uses Qdrant's built-in MaxSim for multi-vector (visual) search and
    standard cosine similarity for all text-based collections.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_prefix: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._prefix = collection_prefix
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = _connect(self._host, self._port)
        return self._client

    def _col(self, name: str) -> str:
        return f"{self._prefix}{name}" if self._prefix else name

    def search_visual(
        self,
        collection: str,
        query_embedding: list[list[float]],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        client = self._get_client()
        hits = client.query_points(
            collection_name=self._col(collection),
            query=query_embedding,
            limit=top_k,
            query_filter=filters,
        ).points
        return [
            RetrievalResult(
                element=_payload_to_element(h.payload),
                score=h.score,
                collection=collection,
                rank=i + 1,
            )
            for i, h in enumerate(hits)
        ]

    def search_text(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        client = self._get_client()
        hits = client.query_points(
            collection_name=self._col(collection),
            query=query_embedding,
            limit=top_k,
            query_filter=filters,
        ).points
        return [
            RetrievalResult(
                element=_payload_to_element(h.payload),
                score=h.score,
                collection=collection,
                rank=i + 1,
            )
            for i, h in enumerate(hits)
        ]


def _payload_to_element(payload: dict) -> Any:
    """Reconstruct an ExtractedElement from a Qdrant point payload."""
    from ...core.models.elements import Figure, Formula, Table, TextChunk, VisualPage

    obj_type = payload.get("object_type", "")
    base = {
        "object_id": payload.get("object_id", ""),
        "doc_id": payload.get("doc_id", ""),
        "source_file": payload.get("source_file", ""),
        "page_number": payload.get("page_number", 0),
        "tool_name": payload.get("tool_name", ""),
        "tool_version": payload.get("tool_version", ""),
    }
    if obj_type == "visual_page":
        return VisualPage(**base, image_path=payload.get("image_path", ""))
    if obj_type == "text_chunk":
        return TextChunk(**base, content=payload.get("content", ""))
    if obj_type == "table":
        return Table(**base, content=payload.get("content", ""))
    if obj_type == "formula":
        return Formula(
            **base,
            latex=payload.get("latex", ""),
            content=payload.get("content", ""),
        )
    if obj_type == "figure":
        return Figure(
            **base,
            image_path=payload.get("image_path", ""),
            description=payload.get("description"),
            caption=payload.get("caption"),
        )
    raise ValueError(f"Unknown object_type in Qdrant payload: '{obj_type}'")
