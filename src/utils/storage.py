from datetime import datetime, timezone
from pathlib import Path


class StorageManager:
    """Manages versioned output directories for pipeline runs.

    Each run gets a timestamped directory so outputs are never silently
    overwritten. This makes it safe to compare runs from different tools.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def run_dir(self, pipeline_name: str, tool_name: str) -> Path:
        """Create and return a new versioned run directory."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{pipeline_name}_{tool_name}_{timestamp}"
        path = self.base_dir / pipeline_name / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def image_path(self, run_dir: Path, doc_id: str, page_number: int) -> Path:
        """Return the canonical path for a rendered page image."""
        return run_dir / f"{doc_id}_p{page_number:04d}.png"

    def element_path(self, run_dir: Path, object_id: str, extension: str) -> Path:
        """Return the path for a specific element's raw output file."""
        return run_dir / f"{object_id}.{extension}"
