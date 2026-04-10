"""Test config loading from YAML and default fallbacks."""

import textwrap
from pathlib import Path

import pytest

from src.core.config import AppConfig, default_config, get_adapter_config, load_config


class TestDefaultConfig:
    def test_returns_app_config(self):
        cfg = default_config()
        assert isinstance(cfg, AppConfig)

    def test_pipeline_defaults(self):
        cfg = default_config()
        assert cfg.pipelines.visual.renderer == "pymupdf"
        assert cfg.pipelines.visual.embedder == "colqwen25"
        assert cfg.pipelines.text.extractor == "olmocr2"
        assert cfg.pipelines.text.chunker == "fixed_size"
        assert cfg.pipelines.structured.parser == "mineru25"

    def test_collection_defaults(self):
        cfg = default_config()
        assert cfg.pipelines.visual.collection == "visual_pages"
        assert cfg.pipelines.text.collection == "text_chunks"
        cols = cfg.pipelines.structured.collections
        assert cols.tables == "tables"
        assert cols.formulas == "formulas"
        assert cols.figures == "figures"


class TestLoadConfig:
    def test_load_minimal_yaml(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            storage:
              base_dir: /data/outputs
            pipelines:
              visual:
                embedder: colpali
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(config_file)
        assert cfg.storage.base_dir == "/data/outputs"
        # Overridden value
        assert cfg.pipelines.visual.embedder == "colpali"
        # Defaults preserved for non-overridden fields
        assert cfg.pipelines.visual.renderer == "pymupdf"
        assert cfg.pipelines.text.extractor == "olmocr2"

    def test_load_adapter_config(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            adapters:
              qdrant:
                host: qdrant.internal
                port: 6334
              bge_m3:
                batch_size: 64
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(config_file)
        qdrant_cfg = get_adapter_config(cfg, "qdrant")
        assert qdrant_cfg["host"] == "qdrant.internal"
        assert qdrant_cfg["port"] == 6334

        bge_cfg = get_adapter_config(cfg, "bge_m3")
        assert bge_cfg["batch_size"] == 64

    def test_load_empty_yaml(self, tmp_path: Path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        cfg = load_config(config_file)
        assert isinstance(cfg, AppConfig)

    def test_missing_adapter_returns_empty_dict(self):
        cfg = default_config()
        assert get_adapter_config(cfg, "nonexistent") == {}
