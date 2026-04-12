import hashlib


def generate_element_id(
    doc_id: str,
    page_number: int,
    object_type: str,
    tool_name: str,
    sequence: int = 0,
) -> str:
    """Generate a stable, deterministic ID for an extracted element.

    Combining all fields ensures uniqueness across documents, pages, tools,
    and multiple elements of the same type on the same page.
    """
    raw = f"{doc_id}:{page_number}:{object_type}:{tool_name}:{sequence}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_doc_id(source_file: str) -> str:
    """Generate a stable document ID from the file contents.

    Hashing content (not path) means the same PDF yields the same ID
    regardless of where it is stored or how it is named.
    """
    h = hashlib.sha256()
    with open(source_file, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
