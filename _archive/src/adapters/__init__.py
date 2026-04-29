# Import all adapter subpackages to trigger their @register_* decorators.
# This must be imported before any registry lookup is performed.
from . import (  # noqa: F401
    chunkers,
    embedders,
    figures,
    formula,
    fusion,
    ocr,
    parsers,
    renderers,
    vectordb,
    visual,
)
