from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> int:
    """Serialize an iterable of Pydantic models to JSONL.

    One model per line, UTF-8, JSON-safe (datetimes → ISO strings via
    `model_dump(mode="json")`). Returns the number of lines written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield each line of a JSONL file as a dict. Streams — no full load."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
