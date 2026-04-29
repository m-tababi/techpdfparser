from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ManifestBuilder:
    """Builds a manifest.json for a single pipeline run.

    Records what ran, which tools were used, how long it took, and
    how many elements were produced. Written to the run directory.
    """

    def __init__(
        self,
        run_id: str,
        pipeline: str,
        doc_id: str,
        source_file: str,
        tools: dict[str, str],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.pipeline = pipeline
        self.doc_id = doc_id
        self.source_file = source_file
        self.tools = tools  # e.g. {"renderer": "pymupdf", "embedder": "clip"}
        self.config = config or {}
        self._started_at = datetime.now(timezone.utc)
        self._tool_versions: dict[str, str] = {}
        self._counts: dict[str, int] = {}
        self._index: dict[str, Any] = {}

    def set_tool_version(self, tool_name: str, version: str) -> None:
        self._tool_versions[tool_name] = version

    def set_counts(self, **kwargs: int) -> None:
        self._counts.update(kwargs)

    def set_index_info(
        self,
        *,
        backend: str,
        namespace: str,
        collections: str | list[str],
        upserted: int,
        adapter_signatures: dict[str, str] | None = None,
        vector_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._index = {
            "backend": backend,
            "namespace": namespace,
            "collections": collections,
            "upserted": upserted,
            "adapter_signatures": adapter_signatures or {},
            "vector_schemas": vector_schemas or {},
        }

    def set_qdrant_info(self, collection: str, upserted: int) -> None:
        self.set_index_info(
            backend="qdrant",
            namespace="legacy",
            collections=collection,
            upserted=upserted,
        )

    def write(self, run_dir: Path) -> None:
        finished_at = datetime.now(timezone.utc)
        duration = (finished_at - self._started_at).total_seconds()
        manifest = {
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "started_at": self._started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round(duration, 3),
            "tools": self.tools,
            "tool_versions": self._tool_versions,
            "config": self.config,
            "counts": self._counts,
            "index": self._index,
        }
        path = run_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def record_tool_version(manifest: ManifestBuilder, adapter: object) -> None:
    """Record adapter version only when both fields are concrete strings."""
    tool_name = getattr(adapter, "tool_name", None)
    tool_version = getattr(adapter, "tool_version", None)
    if isinstance(tool_name, str) and isinstance(tool_version, str):
        manifest.set_tool_version(tool_name, tool_version)
