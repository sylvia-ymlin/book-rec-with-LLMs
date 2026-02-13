"""
Response formatting: converts enriched metadata into API-ready recommendation dicts.
Single responsibility: define the structure of recommendation responses.
"""
from typing import Any, Dict, Union

from src.core.models import BookMetadata, BookResponseDict


def format_book_response(
    meta: BookMetadata | Dict[str, Any], isbn: str, source: str = "local"
) -> BookResponseDict:
    """
    Format a single book's metadata into the standard API response structure.

    Args:
        meta: Enriched metadata dict (from MetadataStore + enrich_book_metadata)
        isbn: ISBN string
        source: Data source label (local, google_books, content_based)

    Returns:
        Dict with isbn, title, authors, description, thumbnail, caption, tags,
        emotions, review_highlights, persona_summary, average_rating, source
    """
    tags_raw = str(meta.get("tags", "")).strip()
    tags = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []

    return {
        "isbn": str(isbn),
        "title": meta.get("title", ""),
        "authors": meta.get("authors", "Unknown"),
        "description": meta.get("description", ""),
        "thumbnail": meta.get("thumbnail") or meta.get("image") or "/content/cover-not-found.jpg",
        "caption": f"{meta.get('title', '')} by {meta.get('authors', 'Unknown')}",
        "tags": tags,
        "emotions": {
            "joy": float(meta.get("joy", 0.0)),
            "sadness": float(meta.get("sadness", 0.0)),
            "fear": float(meta.get("fear", 0.0)),
            "anger": float(meta.get("anger", 0.0)),
            "surprise": float(meta.get("surprise", 0.0)),
        },
        "review_highlights": [
            h.strip()
            for h in str(meta.get("review_highlights", "")).split(";")
            if h.strip()
        ][:3],
        "persona_summary": "",
        "average_rating": float(meta.get("average_rating", 0.0)),
        "source": source,
    }


def format_web_book_response(book: BookMetadata | Dict[str, Any], isbn: str) -> BookResponseDict:
    """
    Format a raw web API book dict into the standard response structure.
    Used when books come from Google Books API (no local metadata).
    """
    return {
        "isbn": isbn,
        "title": book.get("title", ""),
        "authors": book.get("authors", "Unknown"),
        "description": book.get("description", ""),
        "thumbnail": book.get("thumbnail", ""),
        "caption": f"{book.get('title', '')} by {book.get('authors', 'Unknown')}",
        "tags": [],
        "emotions": {"joy": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, "surprise": 0.0},
        "review_highlights": [],
        "persona_summary": "",
        "average_rating": float(book.get("average_rating", 0.0)),
        "source": "google_books",
    }
