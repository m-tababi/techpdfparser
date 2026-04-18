from pathlib import Path

from extraction.config import ExtractionConfig, load_extraction_config


def test_default_config_has_all_fields():
    cfg = ExtractionConfig()
    assert cfg.renderer == "pymupdf"
    assert cfg.segmenter == "mineru25"
    assert cfg.text_extractor == "olmocr2"
    assert cfg.table_extractor == "mineru25"
    assert cfg.formula_extractor == "noop"
    assert cfg.figure_descriptor == "qwen25vl"
    assert cfg.output_dir == "outputs"
    assert cfg.confidence_threshold == 0.3
    assert cfg.dpi == 150


def test_load_from_yaml(tmp_path: Path):
    yaml_content = """
extraction:
  renderer: pymupdf
  segmenter: mineru25
  text_extractor: olmocr2
  output_dir: my_outputs
  confidence_threshold: 0.5

adapters:
  pymupdf:
    dpi: 300
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    cfg = load_extraction_config(config_path)

    assert cfg.output_dir == "my_outputs"
    assert cfg.confidence_threshold == 0.5
    assert cfg.adapters["pymupdf"]["dpi"] == 300


def test_get_adapter_config_returns_empty_for_unknown():
    cfg = ExtractionConfig()
    assert cfg.get_adapter_config("nonexistent") == {}


def test_get_adapter_config_returns_settings():
    cfg = ExtractionConfig(adapters={"pymupdf": {"dpi": 300}})
    assert cfg.get_adapter_config("pymupdf") == {"dpi": 300}
