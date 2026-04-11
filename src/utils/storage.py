from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StorageManager:
    """Manages document-centric output directories for pipeline runs.

    Layout:
        outputs/documents/<doc_id>/
            document.json          # doc metadata + run index
            runs/<run_id>/         # one dir per pipeline run
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def doc_dir(self, doc_id: str) -> Path:
        """Return (and create) the document root directory."""
        path = self.base_dir / "documents" / doc_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_dir(self, doc_id: str, pipeline_name: str, tool_suffix: str) -> Path:
        """Create and return a new versioned run directory.

        Name format: <pipeline>_<tool_suffix>_<YYYYmmdd_HHMMSS>
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{pipeline_name}_{tool_suffix}_{timestamp}"
        path = self.doc_dir(doc_id) / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_id_from_dir(self, run_dir: Path) -> str:
        """Extract the run_id from a run directory path."""
        return run_dir.name

    def image_path(self, run_dir: Path, page_number: int) -> Path:
        """Return the canonical path for a rendered page image."""
        pages_dir = run_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        return pages_dir / f"p{page_number:04d}.png"

    def figure_path(self, run_dir: Path, page_number: int, seq: int) -> Path:
        """Return the canonical path for a figure image inside a run dir."""
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        return figures_dir / f"p{page_number:04d}_f{seq}.png"

    # ------------------------------------------------------------------
    # Document index
    # ------------------------------------------------------------------

    def document_json_path(self, doc_id: str) -> Path:
        return self.doc_dir(doc_id) / "document.json"

    def update_document_index(
        self,
        doc_id: str,
        source_file: str,
        run_id: str,
        pipeline: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append a run entry to document.json, creating it if needed.

        Idempotent on the same run_id — duplicate entries are skipped.
        """
        path = self.document_json_path(doc_id)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"doc_id": doc_id, "source_file": source_file, "runs": []}

        # Skip if this run_id is already recorded
        existing_ids = {r["run_id"] for r in data.get("runs", [])}
        if run_id not in existing_ids:
            entry: dict[str, Any] = {
                "run_id": run_id,
                "pipeline": pipeline,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            if extra:
                entry.update(extra)
            data["runs"].append(entry)

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
