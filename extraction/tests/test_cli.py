"""CLI smoke tests — does argparse wire each subcommand to the right stage?"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from extraction.__main__ import main


def _invoke(*argv: str) -> int:
    with patch.object(sys, "argv", ["extraction", *argv]):
        try:
            main()
            return 0
        except SystemExit as e:
            return int(e.code) if e.code is not None else 0


def test_segment_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(pdfs: list[Path], cfg: object, output_base: Path) -> int:
        calls["args"] = (list(pdfs), output_base)
        return 0

    monkeypatch.setattr("extraction.__main__.run_segment", _fake)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert _invoke("segment", str(pdf), "--out", str(tmp_path / "o")) == 0
    assert calls["args"] == ([pdf], tmp_path / "o")


def test_text_dispatches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def _fake(dirs: list[Path], cfg: object, *, force: bool = False) -> int:
        calls["args"] = (list(dirs), force)
        return 0

    monkeypatch.setattr("extraction.__main__.run_text", _fake)
    d = tmp_path / "d1"
    d.mkdir()
    assert _invoke("extract-text", str(d)) == 0
    assert calls["args"] == ([d], False)


def test_text_dispatches_force(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def _fake(dirs: list[Path], cfg: object, *, force: bool = False) -> int:
        calls["args"] = (list(dirs), force)
        return 0

    monkeypatch.setattr("extraction.__main__.run_text", _fake)
    d = tmp_path / "d1"
    d.mkdir()
    assert _invoke("extract-text", "--force", str(d)) == 0
    assert calls["args"] == ([d], True)


def test_figures_dispatches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def _fake(dirs: list[Path], cfg: object, *, force: bool = False) -> int:
        calls["args"] = (list(dirs), force)
        return 0

    monkeypatch.setattr("extraction.__main__.run_figures", _fake)
    d = tmp_path / "d1"
    d.mkdir()
    assert _invoke("describe-figures", str(d)) == 0
    assert calls["args"] == ([d], False)


def test_figures_dispatches_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def _fake(dirs: list[Path], cfg: object, *, force: bool = False) -> int:
        calls["args"] = (list(dirs), force)
        return 0

    monkeypatch.setattr("extraction.__main__.run_figures", _fake)
    d = tmp_path / "d1"
    d.mkdir()
    assert _invoke("describe-figures", "--force", str(d)) == 0
    assert calls["args"] == ([d], True)


def test_assemble_dispatches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def _fake(dirs: list[Path], cfg: object) -> int:
        calls["args"] = list(dirs)
        return 0

    monkeypatch.setattr("extraction.__main__.run_assemble", _fake)
    d = tmp_path / "d1"
    d.mkdir()
    assert _invoke("assemble", str(d)) == 0
    assert calls["args"] == [d]


def test_unknown_subcommand_exits_nonzero() -> None:
    assert _invoke("nope") in (1, 2)


def test_segment_uses_cfg_output_dir_when_out_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ohne --out greift cfg.output_dir."""
    calls: dict[str, object] = {}

    def _fake(pdfs: list[Path], cfg: object, output_base: Path) -> int:
        calls["out"] = output_base
        return 0

    monkeypatch.setattr("extraction.__main__.run_segment", _fake)

    from extraction.config import ExtractionConfig

    def _fake_cfg(_: object) -> ExtractionConfig:
        return ExtractionConfig(output_dir=str(tmp_path / "from_cfg"))

    monkeypatch.setattr("extraction.__main__._load_cfg", _fake_cfg)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    assert _invoke("segment", str(pdf)) == 0
    assert calls["out"] == tmp_path / "from_cfg"


def test_segment_out_overrides_cfg_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explizites --out hat Vorrang vor cfg.output_dir."""
    calls: dict[str, object] = {}

    def _fake(pdfs: list[Path], cfg: object, output_base: Path) -> int:
        calls["out"] = output_base
        return 0

    monkeypatch.setattr("extraction.__main__.run_segment", _fake)

    from extraction.config import ExtractionConfig

    def _fake_cfg(_: object) -> ExtractionConfig:
        return ExtractionConfig(output_dir=str(tmp_path / "from_cfg"))

    monkeypatch.setattr("extraction.__main__._load_cfg", _fake_cfg)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    assert _invoke("segment", str(pdf), "--out", str(tmp_path / "override")) == 0
    assert calls["out"] == tmp_path / "override"
