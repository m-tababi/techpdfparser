from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class StorageConfig(BaseModel):
    base_dir: str = "outputs"


class VisualPipelineConfig(BaseModel):
    renderer: str = "pymupdf"
    embedder: str = "colqwen25"
    collection: str = "visual_pages"
    dpi: int = 150


class TextPipelineConfig(BaseModel):
    extractor: str = "olmocr2"
    chunker: str = "fixed_size"
    embedder: str = "bge_m3"
    collection: str = "text_chunks"
    chunk_size: int = 512
    chunk_overlap: int = 64


class StructuredCollectionsConfig(BaseModel):
    tables: str = "tables"
    formulas: str = "formulas"
    figures: str = "figures"


class StructuredPipelineConfig(BaseModel):
    parser: str = "mineru25"
    formula_extractor: str = "ppformulanet"
    figure_descriptor: str = "qwen25vl"
    collections: StructuredCollectionsConfig = Field(
        default_factory=StructuredCollectionsConfig
    )


class PipelinesConfig(BaseModel):
    visual: VisualPipelineConfig = Field(default_factory=VisualPipelineConfig)
    text: TextPipelineConfig = Field(default_factory=TextPipelineConfig)
    structured: StructuredPipelineConfig = Field(default_factory=StructuredPipelineConfig)


class RetrievalConfig(BaseModel):
    retrieval_engine: str = "qdrant"
    fusion_engine: str = "rrf"
    index_namespace: str = "auto"
    validate_on_start: bool = True
    fail_on_schema_mismatch: bool = True


class AppConfig(BaseModel):
    """Root config object. Load via `load_config()` or use `default_config()`."""

    storage: StorageConfig = Field(default_factory=StorageConfig)
    pipelines: PipelinesConfig = Field(default_factory=PipelinesConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    # Per-adapter settings keyed by adapter name (e.g. "colqwen25", "qdrant")
    adapters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_chunker_dependencies(self) -> "AppConfig":
        if (
            self.pipelines.text.chunker == "section_aware"
            and self.pipelines.text.extractor != "pymupdf_structured"
        ):
            raise ValueError(
                "The 'section_aware' chunker requires an extractor that emits "
                "section metadata. Use 'pymupdf_structured' for "
                "pipelines.text.extractor."
            )
        return self


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file into AppConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw or {})


_DEFAULT_CONFIG_PATH = Path("config.yaml")


def default_config() -> AppConfig:
    """Load config.yaml from the current directory if present, else use built-in defaults."""
    if _DEFAULT_CONFIG_PATH.exists():
        return load_config(_DEFAULT_CONFIG_PATH)
    return AppConfig()


def get_adapter_config(app_config: AppConfig, adapter_name: str) -> dict[str, Any]:
    """Extract per-adapter settings from the root config.

    Returns an empty dict if the adapter has no dedicated config block,
    so adapters can always call this safely and fall back to their defaults.
    """
    return app_config.adapters.get(adapter_name, {})
