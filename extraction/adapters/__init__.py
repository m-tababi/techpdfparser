"""Import all adapter modules to trigger @register_* decorators.

Heavy adapters (MinerU, transformers-based extractors) are wrapped in
try/except so the module imports cleanly even without their Python
deps installed — the registry entry is created by the decorator at
class-body time, which happens only when the module import succeeds.
"""
import logging

log = logging.getLogger(__name__)

from . import pymupdf_renderer, pymupdf_text_segmenter, stubs  # noqa: F401, E402

try:
    from . import mineru25_segmenter  # noqa: F401
except ImportError as exc:
    log.debug("mineru25_segmenter not registered: %s", exc)

try:
    from . import olmocr2_text  # noqa: F401
except ImportError as exc:
    log.debug("olmocr2_text not registered: %s", exc)

try:
    from . import qwen25vl_figure  # noqa: F401
except ImportError as exc:
    log.debug("qwen25vl_figure not registered: %s", exc)

try:
    from . import qwen25vl_table  # noqa: F401
except ImportError as exc:
    log.debug("qwen25vl_table not registered: %s", exc)

try:
    from . import tatr_table  # noqa: F401
except ImportError as exc:
    log.debug("tatr_table not registered: %s", exc)

try:
    from . import docling_table  # noqa: F401
except ImportError as exc:
    log.debug("docling_table not registered: %s", exc)
