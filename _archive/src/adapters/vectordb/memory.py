from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, cast

from ...core.indexing import VectorSchema, schema_matches
from ...core.models.elements import Figure, Formula, Table, TextChunk, VisualPage
from ...core.models.results import RetrievalResult
from ...core.registry import register_index_writer, register_retrieval_engine
from .qdrant import _base_payload, _payload_to_element

Vector = list[float]
MultiVector = list[Vector]


@dataclass
class _StoredPoint:
    vector: Vector | MultiVector
    payload: dict[str, Any]


@dataclass
class _MemoryCollection:
    schema: VectorSchema
    points: dict[str, _StoredPoint] = field(default_factory=dict)


@dataclass
class _MemoryStore:
    collections: dict[str, _MemoryCollection] = field(default_factory=dict)


_STORES: dict[str, _MemoryStore] = {}


def _get_store(name: str) -> _MemoryStore:
    return _STORES.setdefault(name, _MemoryStore())


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _maxsim(query: list[list[float]], doc: list[list[float]]) -> float:
    if not query or not doc:
        return 0.0
    return sum(max(_cosine(q, d) for d in doc) for q in query)


def _as_vector(value: Vector | MultiVector) -> Vector:
    if value and isinstance(value[0], list):
        raise TypeError("Expected a single-vector payload")
    return cast(Vector, value)


def _as_multivector(value: Vector | MultiVector) -> MultiVector:
    if not value:
        return []
    if not isinstance(value[0], list):
        raise TypeError("Expected a multi-vector payload")
    return cast(MultiVector, value)


class _MemoryBackend:
    def __init__(
        self,
        store_name: str = "default",
        collection_prefix: str = "",
    ) -> None:
        self._store_name = store_name
        self._prefix = collection_prefix

    @property
    def store(self) -> _MemoryStore:
        return _get_store(self._store_name)

    def _col(self, name: str) -> str:
        return f"{self._prefix}{name}" if self._prefix else name


@register_index_writer("memory")
class MemoryIndexWriter(_MemoryBackend):
    def healthcheck(self) -> None:
        return None

    def get_collection_schema(self, collection: str) -> VectorSchema | None:
        current = self.store.collections.get(self._col(collection))
        return current.schema if current else None

    def ensure_collection(
        self,
        collection: str,
        schema: VectorSchema,
        fail_on_schema_mismatch: bool = True,
    ) -> None:
        col = self._col(collection)
        current = self.store.collections.get(col)
        if current is None:
            self.store.collections[col] = _MemoryCollection(schema=schema)
            return
        if not schema_matches(schema, current.schema) and fail_on_schema_mismatch:
            raise ValueError(
                f"Collection '{col}' has schema {current.schema.model_dump()} "
                f"but expected {schema.model_dump()}"
            )

    def _upsert(
        self,
        collection: str,
        elements: list[Any],
        *,
        vector_getter: str,
        payload_builder: Callable[[Any], dict[str, Any]],
    ) -> None:
        current = self.store.collections[self._col(collection)]
        for element in elements:
            vector = getattr(element, vector_getter)
            if not vector:
                continue
            current.points[element.object_id] = _StoredPoint(
                vector=vector,
                payload=payload_builder(element),
            )

    def upsert_visual(self, collection: str, pages: list[VisualPage]) -> None:
        self._upsert(
            collection,
            pages,
            vector_getter="embedding",
            payload_builder=lambda page: {
                **_base_payload(page),
                "image_path": page.image_path,
            },
        )

    def upsert_text(self, collection: str, chunks: list[TextChunk]) -> None:
        self._upsert(
            collection,
            chunks,
            vector_getter="embedding",
            payload_builder=lambda chunk: {
                **_base_payload(chunk),
                "content": chunk.content,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            },
        )

    def upsert_tables(self, collection: str, tables: list[Table]) -> None:
        self._upsert(
            collection,
            tables,
            vector_getter="embedding",
            payload_builder=lambda table: {
                **_base_payload(table),
                "content": table.content,
            },
        )

    def upsert_formulas(self, collection: str, formulas: list[Formula]) -> None:
        self._upsert(
            collection,
            formulas,
            vector_getter="embedding",
            payload_builder=lambda formula: {
                **_base_payload(formula),
                "latex": formula.latex,
                "content": formula.content,
                "image_path": formula.image_path,
            },
        )

    def upsert_figures(self, collection: str, figures: list[Figure]) -> None:
        self._upsert(
            collection,
            figures,
            vector_getter="embedding",
            payload_builder=lambda fig: {
                **_base_payload(fig),
                "image_path": fig.image_path,
                "description": fig.description,
                "caption": fig.caption,
            },
        )


@register_retrieval_engine("memory")
class MemoryRetrievalEngine(_MemoryBackend):
    def _rank(
        self,
        collection: str,
        scores: list[tuple[str, float]],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        current = self.store.collections.get(self._col(collection))
        if current is None:
            return []
        ordered = sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievalResult(
                element=_payload_to_element(current.points[object_id].payload),
                score=score,
                collection=collection,
                rank=index + 1,
            )
            for index, (object_id, score) in enumerate(ordered)
        ]

    def search_visual(
        self,
        collection: str,
        query_embedding: list[list[float]],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        del filters
        current = self.store.collections.get(self._col(collection))
        if current is None:
            return []
        scores = [
            (object_id, _maxsim(query_embedding, _as_multivector(point.vector)))
            for object_id, point in current.points.items()
        ]
        return self._rank(collection, scores, top_k=top_k)

    def search_text(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        del filters
        current = self.store.collections.get(self._col(collection))
        if current is None:
            return []
        scores = [
            (object_id, _cosine(query_embedding, _as_vector(point.vector)))
            for object_id, point in current.points.items()
        ]
        return self._rank(collection, scores, top_k=top_k)
