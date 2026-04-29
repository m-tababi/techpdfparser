from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.models.elements import BaseElement


@dataclass
class SectionMarker:
    page: int
    y0: float
    level: int  # 1-based
    title: str
    path: list[str] = field(default_factory=list)  # full hierarchy, e.g. ["2 Methods", "2.3 Data"]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_sections_from_toc(doc: object) -> list[SectionMarker]:
    """Extract section markers from a PyMuPDF document outline (TOC).

    Returns an empty list when the PDF has no embedded outline.
    TOC entries: [level, title, page, dest] — page is 1-based in fitz.
    """
    toc: list[tuple[int, str, int]] = doc.get_toc()  # type: ignore[attr-defined]
    if not toc:
        return []

    markers: list[SectionMarker] = []
    # Track current path per level to build cumulative hierarchy
    path_by_level: dict[int, str] = {}

    for entry in toc:
        level, title, page_1based = entry[0], entry[1], entry[2]
        title = title.strip()
        if not title:
            continue
        page = max(0, page_1based - 1)  # convert to 0-based

        path_by_level[level] = title
        # Remove deeper levels that no longer apply
        for deeper in list(path_by_level):
            if deeper > level:
                del path_by_level[deeper]

        path = [path_by_level[lvl] for lvl in sorted(path_by_level)]
        # y0=0 — TOC has no y-coordinate; assign_sections uses page order
        markers.append(SectionMarker(page=page, y0=0.0, level=level, title=title, path=list(path)))

    return markers


def detect_sections_from_fonts(
    spans: list[dict],
    size_ratio: float = 1.3,
    max_levels: int = 4,
) -> list[SectionMarker]:
    """Heuristic heading detection via font size and bold flag.

    Spans come from page.get_text("dict") → block → line → span.
    Bold flag is bit 16 (fitz flags field).

    Strategy: collect all span sizes → median = body text.
    Spans >= size_ratio * median OR bold are candidates.
    Bucket distinct sizes into up to max_levels levels (largest = level 1).
    """
    sizes = [s["size"] for s in spans if s.get("text", "").strip()]
    if not sizes:
        return []

    body_size = statistics.median(sizes)
    heading_threshold = body_size * size_ratio

    # Collect unique heading sizes for bucketing
    heading_sizes: set[float] = set()
    for span in spans:
        text = span.get("text", "").strip()
        if not text:
            continue
        size = span["size"]
        is_bold = bool(span.get("flags", 0) & 16)
        if size >= heading_threshold or is_bold:
            heading_sizes.add(round(size, 1))

    if not heading_sizes:
        return []

    # Map sizes to levels: largest size = level 1
    sorted_sizes = sorted(heading_sizes, reverse=True)[:max_levels]
    size_to_level = {s: i + 1 for i, s in enumerate(sorted_sizes)}

    markers: list[SectionMarker] = []
    path_by_level: dict[int, str] = {}

    for span in spans:
        text = span.get("text", "").strip()
        if not text:
            continue
        size = round(span["size"], 1)
        is_bold = bool(span.get("flags", 0) & 16)
        if size not in size_to_level and not is_bold:
            continue
        # Bold spans not in size_to_level get appended at deepest level
        level = size_to_level.get(size, max_levels)
        page = span.get("page", 0)
        y0 = span.get("origin", (0, 0))[1]

        path_by_level[level] = text
        for deeper in list(path_by_level):
            if deeper > level:
                del path_by_level[deeper]
        path = [path_by_level[lvl] for lvl in sorted(path_by_level)]

        markers.append(SectionMarker(page=page, y0=y0, level=level, title=text, path=list(path)))

    return markers


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def assign_sections(blocks: list[BaseElement], markers: list[SectionMarker]) -> None:
    """Set section_title / section_path / heading_level on blocks in-place.

    Blocks inherit the most-recently-passed marker based on (page, y0) order.
    Markers from TOC have y0=0, so they are passed at the top of each page.
    """
    if not markers:
        return

    # Sort markers by (page, y0) for sequential scan
    ordered = sorted(markers, key=lambda m: (m.page, m.y0))
    current: SectionMarker | None = None
    marker_idx = 0

    for block in sorted(blocks, key=lambda b: (b.page_number, b.bbox.y0 if b.bbox else 0.0)):
        block_page = block.page_number
        block_y = block.bbox.y0 if block.bbox else 0.0

        # Advance marker pointer past all markers that precede this block
        while marker_idx < len(ordered):
            m = ordered[marker_idx]
            if (m.page, m.y0) <= (block_page, block_y):
                current = m
                marker_idx += 1
            else:
                break

        if current is not None:
            block.section_title = current.title
            block.section_path = list(current.path)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_sections(path: Path, markers: list[SectionMarker]) -> None:
    """Serialize markers to JSON for cross-pipeline linkage."""
    data = [
        {"page": m.page, "y0": m.y0, "level": m.level, "title": m.title, "path": m.path}
        for m in markers
    ]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_sections(path: Path) -> list[SectionMarker]:
    """Deserialize markers from a sections.json file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        SectionMarker(
            page=entry["page"],
            y0=entry["y0"],
            level=entry["level"],
            title=entry["title"],
            path=entry["path"],
        )
        for entry in data
    ]
