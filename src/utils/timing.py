import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class TimingResult:
    name: str
    elapsed_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


@contextmanager
def timed(name: str, metadata: dict | None = None) -> Generator[TimingResult, None, None]:
    """Context manager that measures wall-clock time for a named operation.

    Usage:
        with timed("embed_batch") as t:
            embeddings = model.encode(texts)
        print(f"Took {t.elapsed_seconds:.2f}s")
    """
    result = TimingResult(name=name, metadata=metadata or {})
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start
