from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


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


class AppConfig(BaseModel):
    """Root config object. Load via `load_config()` or use `default_config()`."""

    storage: StorageConfig = Field(default_factory=StorageConfig)
    pipelines: PipelinesConfig = Field(default_factory=PipelinesConfig)
    # Per-adapter settings keyed by adapter name (e.g. "colqwen25", "qdrant")
    adapters: dict[str, dict[str, Any]] = Field(default_factory=dict)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file into AppConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw or {})


def default_config() -> AppConfig:
    """Return a default AppConfig for testing or quick-start use."""
    return AppConfig()


def get_adapter_config(app_config: AppConfig, adapter_name: str) -> dict[str, Any]:
    """Extract per-adapter settings from the root config.

    Returns an empty dict if the adapter has no dedicated config block,
    so adapters can always call this safely and fall back to their defaults.
    """
    return app_config.adapters.get(adapter_name, {})
