"""Extraction block configuration.

Load from YAML or use defaults. Each adapter has its own config section.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_OUTPUT_BASE = "outputs"


class ExtractionConfig(BaseModel):
    """Configuration for the extraction pipeline."""

    renderer: str = "pymupdf"
    segmenter: str = "mineru_vlm"
    # MinerU's middle_json already carries per-region text and LaTeX; the
    # passthrough extractors let the pipeline keep that content via role-match
    # instead of re-running an OCR/formula model. olmocr2 stays available as
    # an alternative for image-only PDFs.
    text_extractor: str = "mineru_vlm"
    table_extractor: str = "qwen25vl_table"
    formula_extractor: str = "mineru_vlm"
    figure_descriptor: str = "tatr"
    output_dir: str = "outputs"
    confidence_threshold: float = 0.3
    dpi: int = 150
    adapters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_adapter_config(self, adapter_name: str) -> dict[str, Any]:
        return self.adapters.get(adapter_name, {})

    def resolve_renderer_dpi(self) -> int:
        """Return effective DPI: adapter block overrides top-level dpi if set."""
        adapter_cfg = self.get_adapter_config(self.renderer)
        if "dpi" in adapter_cfg:
            return int(adapter_cfg["dpi"])
        return int(self.dpi)


def load_extraction_config(path: str | Path) -> ExtractionConfig:
    """Load extraction config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    extraction_raw = raw.get("extraction", {})
    extraction_raw["adapters"] = raw.get("adapters", {})
    return ExtractionConfig.model_validate(extraction_raw)
