from __future__ import annotations

from pydantic import BaseModel, Field

from .elements import ExtractedElement


class RetrievalResult(BaseModel):
    """A single result from a vector search query."""

    element: ExtractedElement
    score: float
    collection: str
    rank: int | None = None


class FusionResult(BaseModel):
    """A merged result produced by a FusionEngine across multiple collections."""

    element: ExtractedElement
    fused_score: float
    # Per-collection raw scores before fusion, for analysis and debugging
    source_scores: dict[str, float] = Field(default_factory=dict)
    rank: int | None = None


class BenchmarkEntry(BaseModel):
    """Timing and resource metrics for one pipeline run on one document."""

    pipeline_name: str
    tool_name: str
    tool_version: str
    doc_id: str
    total_elements: int
    elapsed_seconds: float
    peak_memory_mb: float | None = None
    storage_bytes: int | None = None
    extra: dict = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    """Aggregated results from a benchmarking session."""

    entries: list[BenchmarkEntry] = Field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Return per-tool average latency for quick A/B comparison."""
        totals: dict[str, list[float]] = {}
        for entry in self.entries:
            totals.setdefault(entry.tool_name, []).append(entry.elapsed_seconds)
        return {tool: sum(ts) / len(ts) for tool, ts in totals.items()}
