from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field

from .config import AppConfig


class VectorSchema(BaseModel):
    """Backend-agnostic vector collection contract."""

    dim: int
    distance: str = "cosine"
    multi_vector: bool = False


class ResolvedIndexLayout(BaseModel):
    """Resolved backend, namespace, collection names, and schemas."""

    backend: str
    namespace: str = ""
    adapter_signatures: dict[str, str] = Field(default_factory=dict)
    collections: dict[str, str] = Field(default_factory=dict)
    vector_schemas: dict[str, VectorSchema] = Field(default_factory=dict)


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_distance(distance: str) -> str:
    return distance.strip().lower()


def slugify_component(value: str) -> str:
    cleaned = _NON_ALNUM.sub("-", value.strip().lower()).strip("-")
    return cleaned or "default"


def build_adapter_signature(
    *,
    tool_name: str,
    model_name: str,
    schema: VectorSchema,
) -> str:
    mode = "mv" if schema.multi_vector else "sv"
    model_slug = slugify_component(model_name.split("/")[-1])[:24]
    raw = (
        f"{tool_name}|{model_name}|{schema.dim}|"
        f"{normalize_distance(schema.distance)}|{mode}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{slugify_component(tool_name)}-{model_slug}-{schema.dim}d-{mode}-{digest}"


def get_model_name(adapter: object) -> str:
    value = getattr(adapter, "model_name", None)
    if isinstance(value, str) and value:
        return value

    value = getattr(adapter, "_model_name", None)
    if isinstance(value, str) and value:
        return value

    tool_version = getattr(adapter, "tool_version", None)
    if isinstance(tool_version, str) and tool_version:
        return tool_version

    tool_name = getattr(adapter, "tool_name", None)
    if isinstance(tool_name, str) and tool_name:
        return tool_name

    return adapter.__class__.__name__


def get_text_vector_schema(adapter: object) -> VectorSchema:
    value = getattr(adapter, "vector_schema", None)
    if isinstance(value, VectorSchema):
        return value

    dim = getattr(adapter, "embedding_dim", None)
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(
            f"Text embedder {adapter.__class__.__name__} does not expose a valid embedding_dim"
        )
    return VectorSchema(dim=dim, distance="cosine", multi_vector=False)


def get_visual_vector_schema(adapter: object) -> VectorSchema:
    value = getattr(adapter, "vector_schema", None)
    if isinstance(value, VectorSchema):
        return value

    dim = getattr(adapter, "embedding_dim", None)
    is_multi_vector = getattr(adapter, "is_multi_vector", None)
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(
            f"Visual embedder {adapter.__class__.__name__} does not expose a valid embedding_dim"
        )
    if not isinstance(is_multi_vector, bool):
        raise ValueError(
            f"Visual embedder {adapter.__class__.__name__} does not expose is_multi_vector"
        )
    return VectorSchema(dim=dim, distance="cosine", multi_vector=is_multi_vector)


def get_embedder_signature(adapter: object, schema: VectorSchema | None = None) -> str:
    value = getattr(adapter, "adapter_signature", None)
    if isinstance(value, str) and value:
        return value

    if schema is None:
        is_multi_vector = getattr(adapter, "is_multi_vector", None)
        schema = (
            get_visual_vector_schema(adapter)
            if isinstance(is_multi_vector, bool)
            else get_text_vector_schema(adapter)
        )

    tool_name = getattr(adapter, "tool_name", adapter.__class__.__name__)
    if not isinstance(tool_name, str):
        tool_name = adapter.__class__.__name__

    return build_adapter_signature(
        tool_name=tool_name,
        model_name=get_model_name(adapter),
        schema=schema,
    )


def resolve_namespace(
    namespace_setting: str,
    *,
    backend: str,
    visual_signature: str,
    text_signature: str,
) -> str:
    setting = namespace_setting.strip()
    if setting == "legacy":
        return ""
    if setting and setting != "auto":
        return slugify_component(setting)
    parts = (
        slugify_component(backend),
        slugify_component(visual_signature),
        slugify_component(text_signature),
    )
    return "__".join(parts)


def apply_namespace(base_collection: str, namespace: str) -> str:
    if not namespace:
        return base_collection
    return f"{namespace}__{base_collection}"


def resolve_index_layout(
    config: AppConfig,
    *,
    visual_embedder: object,
    text_embedder: object,
) -> ResolvedIndexLayout:
    visual_schema = get_visual_vector_schema(visual_embedder)
    text_schema = get_text_vector_schema(text_embedder)

    visual_signature = get_embedder_signature(visual_embedder, visual_schema)
    text_signature = get_embedder_signature(text_embedder, text_schema)

    namespace = resolve_namespace(
        config.retrieval.index_namespace,
        backend=config.retrieval.retrieval_engine,
        visual_signature=visual_signature,
        text_signature=text_signature,
    )

    collections = {
        "visual": apply_namespace(config.pipelines.visual.collection, namespace),
        "text": apply_namespace(config.pipelines.text.collection, namespace),
        "tables": apply_namespace(
            config.pipelines.structured.collections.tables, namespace
        ),
        "formulas": apply_namespace(
            config.pipelines.structured.collections.formulas, namespace
        ),
        "figures": apply_namespace(
            config.pipelines.structured.collections.figures, namespace
        ),
    }

    return ResolvedIndexLayout(
        backend=config.retrieval.retrieval_engine,
        namespace=namespace,
        adapter_signatures={
            "visual_embedder": visual_signature,
            "text_embedder": text_signature,
        },
        collections=collections,
        vector_schemas={
            "visual": visual_schema,
            "text": text_schema,
            "tables": text_schema,
            "formulas": text_schema,
            "figures": text_schema,
        },
    )


def schema_matches(expected: VectorSchema, actual: VectorSchema) -> bool:
    return (
        expected.dim == actual.dim
        and normalize_distance(expected.distance) == normalize_distance(actual.distance)
        and expected.multi_vector == actual.multi_vector
    )


def layout_metadata(layout: ResolvedIndexLayout) -> dict[str, Any]:
    return {
        "backend": layout.backend,
        "namespace": layout.namespace or "legacy",
        "adapter_signatures": dict(layout.adapter_signatures),
        "collections": dict(layout.collections),
        "vector_schemas": {
            key: schema.model_dump(mode="json")
            for key, schema in layout.vector_schemas.items()
        },
    }
