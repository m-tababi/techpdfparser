import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging for all pipeline components.

    Uses a consistent format so logs can be parsed by Loki, Datadog, or grep.
    Call once at application startup.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger("techpdfparser")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    return root


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger under the techpdfparser namespace."""
    return logging.getLogger(f"techpdfparser.{name}")
