from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models.results import BenchmarkReport


class BenchmarkRunner(Protocol):
    """Runs a pipeline against a set of documents and records metrics.

    Enables A/B comparisons: run the same inputs through two different
    pipeline configurations and compare the BenchmarkReport outputs.
    """

    def run(
        self,
        pipeline_name: str,
        inputs: list[Path],
        repeat: int = 1,
    ) -> BenchmarkReport:
        """Run the pipeline on all input PDFs and aggregate metrics.

        `repeat` runs each document that many times to average out variance.
        """
        ...
