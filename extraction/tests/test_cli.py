import subprocess
import sys


def test_cli_shows_usage_without_args():
    result = subprocess.run(
        [sys.executable, "-m", "extraction"],
        capture_output=True,
        text=True,
    )
    assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()


def test_cli_extract_requires_pdf_arg():
    result = subprocess.run(
        [sys.executable, "-m", "extraction", "extract"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
