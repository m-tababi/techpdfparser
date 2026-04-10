"""Adapter registry for config-driven tool switching.

Each adapter registers itself by name using a decorator:

    @register_renderer("pymupdf")
    class PyMuPDFRenderer: ...

The pipeline factories then call `get_renderer("pymupdf", **kwargs)`
to instantiate the right class based on the YAML config.
"""

from __future__ import annotations

from typing import Any

_RENDERERS: dict[str, type] = {}
_VISUAL_EMBEDDERS: dict[str, type] = {}
_TEXT_EXTRACTORS: dict[str, type] = {}
_TEXT_CHUNKERS: dict[str, type] = {}
_TEXT_EMBEDDERS: dict[str, type] = {}
_STRUCTURED_PARSERS: dict[str, type] = {}
_FORMULA_EXTRACTORS: dict[str, type] = {}
_FIGURE_DESCRIPTORS: dict[str, type] = {}
_INDEX_WRITERS: dict[str, type] = {}
_RETRIEVAL_ENGINES: dict[str, type] = {}
_FUSION_ENGINES: dict[str, type] = {}


def _make_register(registry: dict[str, type]):
    """Factory: returns a class decorator that inserts into `registry`."""

    def register(name: str):
        def decorator(cls: type) -> type:
            registry[name] = cls
            return cls

        return decorator

    return register


def _make_get(registry: dict[str, type], category: str):
    """Factory: returns a lookup function that instantiates from `registry`."""

    def get(name: str, **kwargs: Any) -> Any:
        if name not in registry:
            available = sorted(registry.keys())
            raise KeyError(
                f"Unknown {category} adapter '{name}'. Available: {available}"
            )
        return registry[name](**kwargs)

    return get


register_renderer = _make_register(_RENDERERS)
register_visual_embedder = _make_register(_VISUAL_EMBEDDERS)
register_text_extractor = _make_register(_TEXT_EXTRACTORS)
register_text_chunker = _make_register(_TEXT_CHUNKERS)
register_text_embedder = _make_register(_TEXT_EMBEDDERS)
register_structured_parser = _make_register(_STRUCTURED_PARSERS)
register_formula_extractor = _make_register(_FORMULA_EXTRACTORS)
register_figure_descriptor = _make_register(_FIGURE_DESCRIPTORS)
register_index_writer = _make_register(_INDEX_WRITERS)
register_retrieval_engine = _make_register(_RETRIEVAL_ENGINES)
register_fusion_engine = _make_register(_FUSION_ENGINES)

get_renderer = _make_get(_RENDERERS, "renderer")
get_visual_embedder = _make_get(_VISUAL_EMBEDDERS, "visual_embedder")
get_text_extractor = _make_get(_TEXT_EXTRACTORS, "text_extractor")
get_text_chunker = _make_get(_TEXT_CHUNKERS, "text_chunker")
get_text_embedder = _make_get(_TEXT_EMBEDDERS, "text_embedder")
get_structured_parser = _make_get(_STRUCTURED_PARSERS, "structured_parser")
get_formula_extractor = _make_get(_FORMULA_EXTRACTORS, "formula_extractor")
get_figure_descriptor = _make_get(_FIGURE_DESCRIPTORS, "figure_descriptor")
get_index_writer = _make_get(_INDEX_WRITERS, "index_writer")
get_retrieval_engine = _make_get(_RETRIEVAL_ENGINES, "retrieval_engine")
get_fusion_engine = _make_get(_FUSION_ENGINES, "fusion_engine")
