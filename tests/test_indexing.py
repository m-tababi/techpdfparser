"""Tests for namespace and schema resolution."""

from src.core.config import AppConfig
from src.core.indexing import VectorSchema, resolve_index_layout


class _VisualEmbedder:
    tool_name = "clip"
    tool_version = "1.0"
    embedding_dim = 512
    is_multi_vector = True
    model_name = "openai/clip-vit-base-patch32"
    vector_schema = VectorSchema(dim=512, distance="cosine", multi_vector=True)
    adapter_signature = "clip-vit-base-patch32-512d-mv-deadbeef"


class _TextEmbedder:
    tool_name = "minilm"
    tool_version = "1.0"
    embedding_dim = 384
    model_name = "all-MiniLM-L6-v2"
    vector_schema = VectorSchema(dim=384, distance="cosine", multi_vector=False)
    adapter_signature = "minilm-all-minilm-l6-v2-384d-sv-cafebabe"


def test_auto_namespace_uses_backend_and_signatures():
    cfg = AppConfig()
    cfg.retrieval.retrieval_engine = "memory"

    layout = resolve_index_layout(
        cfg,
        visual_embedder=_VisualEmbedder(),
        text_embedder=_TextEmbedder(),
    )

    assert layout.namespace.startswith("memory__clip-vit-base-patch32")
    assert layout.collections["text"].startswith(layout.namespace)


def test_legacy_namespace_keeps_base_collection_names():
    cfg = AppConfig()
    cfg.retrieval.index_namespace = "legacy"

    layout = resolve_index_layout(
        cfg,
        visual_embedder=_VisualEmbedder(),
        text_embedder=_TextEmbedder(),
    )

    assert layout.namespace == ""
    assert layout.collections["visual"] == "visual_pages"
    assert layout.collections["text"] == "text_chunks"


def test_changing_text_embedder_changes_namespace():
    cfg = AppConfig()
    cfg.retrieval.retrieval_engine = "memory"

    layout_a = resolve_index_layout(
        cfg,
        visual_embedder=_VisualEmbedder(),
        text_embedder=_TextEmbedder(),
    )

    other_text = _TextEmbedder()
    other_text.adapter_signature = "bge-m3-baai-bge-m3-1024d-sv-facefeed"
    other_text.vector_schema = VectorSchema(
        dim=1024, distance="cosine", multi_vector=False
    )
    layout_b = resolve_index_layout(
        cfg,
        visual_embedder=_VisualEmbedder(),
        text_embedder=other_text,
    )

    assert layout_a.namespace != layout_b.namespace
