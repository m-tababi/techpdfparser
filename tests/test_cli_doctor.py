"""Tests for the doctor CLI."""

from __future__ import annotations

import sys
import textwrap

import pytest

from src import __main__ as cli
from src.adapters.vectordb.memory import MemoryIndexWriter
from src.core.indexing import VectorSchema


def _write_config(tmp_path, *, store_name: str, extra: str = ""):
    config = textwrap.dedent(
        f"""\
        retrieval:
          retrieval_engine: memory
          index_namespace: auto
        pipelines:
          visual:
            embedder: clip
          text:
            extractor: pymupdf_structured
            chunker: section_aware
            embedder: minilm
          structured:
            parser: pdfplumber
            formula_extractor: pix2tex
            figure_descriptor: moondream
        adapters:
          memory:
            store_name: {store_name}
        {extra}
        """
    )
    path = tmp_path / "config.yaml"
    path.write_text(config)
    return path


def test_doctor_succeeds_with_memory_backend(monkeypatch, capsys, tmp_path):
    config_path = _write_config(tmp_path, store_name="doctor_ok")
    monkeypatch.setattr(cli, "_probe_dependencies", lambda cfg: [])
    monkeypatch.setattr(sys, "argv", ["src", "doctor", "--config", str(config_path)])

    cli.main()

    output = capsys.readouterr().out
    assert "Status: OK" in output


def test_doctor_fails_on_schema_mismatch(monkeypatch, capsys, tmp_path):
    config_path = _write_config(tmp_path, store_name="doctor_bad")
    monkeypatch.setattr(cli, "_probe_dependencies", lambda cfg: [])
    cfg = cli._load_cfg(config_path)
    runtime = cli._build_runtime(cfg)

    writer = MemoryIndexWriter(store_name="doctor_bad")
    writer.ensure_collection(
        runtime.index_layout.collections["text"],
        VectorSchema(dim=99, distance="cosine", multi_vector=False),
    )

    monkeypatch.setattr(sys, "argv", ["src", "doctor", "--config", str(config_path)])
    with pytest.raises(SystemExit, match="1"):
        cli.main()

    output = capsys.readouterr().out
    assert "FAILED" in output
