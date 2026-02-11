"""
Centralized ISBN extraction from various document formats.
Single place for robust ISBN parsing logic — used by recommender, agentic nodes, etc.
"""
from typing import Any, Optional


def extract_isbn(doc: Any) -> Optional[str]:
    """
    Extract ISBN from a document (LangChain Document, vector search result, etc.).

    Tries, in order:
    1. metadata['isbn'] or metadata['isbn13']
    2. Content format "Title... ISBN: X"
    3. Legacy format: first token of page_content

    Args:
        doc: Object with .metadata and/or .page_content attributes

    Returns:
        ISBN string if found, None otherwise
    """
    isbn_str: Optional[str] = None

    # 1. Try metadata (Hybrid/BM25)
    if hasattr(doc, "metadata") and doc.metadata:
        if "isbn" in doc.metadata:
            isbn_str = str(doc.metadata["isbn"])
        elif "isbn13" in doc.metadata:
            isbn_str = str(doc.metadata["isbn13"])

    # 2. Try content format "Title... ISBN: X"
    if not isbn_str and hasattr(doc, "page_content") and doc.page_content and "ISBN:" in doc.page_content:
        try:
            parts = doc.page_content.split("ISBN:")
            if len(parts) > 1:
                isbn_str = parts[1].strip().split()[0]
        except (IndexError, AttributeError):
            pass

    # 3. Legacy: first token of page_content
    if not isbn_str and hasattr(doc, "page_content") and doc.page_content:
        isbn_str = doc.page_content.strip('"').split()[0] if doc.page_content.strip() else None

    return isbn_str.strip() if (isbn_str and isbn_str.strip()) else None
