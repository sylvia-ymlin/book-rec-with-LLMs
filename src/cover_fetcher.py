"""
Real-time book cover fetcher using Google Books API.
Falls back to Open Library if Google Books doesn't have the cover.

This module provides dynamic book cover fetching to replace hardcoded file paths
in the dataset. It supports:
- Primary source: Google Books API (isbn search)
- Fallback: Open Library Cover API
- LRU caching to minimize redundant API calls
- Graceful degradation with placeholder images

Performance:
- ~50-200ms per book (with caching: ~0ms for repeated queries)
- 10 books recommendation: ~0.5-1s additional latency
- Cache size: 1000 most recent books

API Rate Limits:
- Google Books: No explicit limit for free tier, but rate-limited
- Open Library: No authentication required

Author: Modified 2026-01-06
"""
import requests
from pathlib import Path
import time
from functools import lru_cache

# Placeholder image for books without covers (local asset)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_COVER = str(PROJECT_ROOT / "assets" / "cover-not-found.jpg")

@lru_cache(maxsize=1000)
def fetch_book_cover(isbn: str, title: str = "") -> tuple[str, str]:
    """
    Fetch book cover URL (Google Books -> Open Library) and best-effort authors.

    Returns:
        (cover_url, authors_str)
    """
    cover = PLACEHOLDER_COVER
    authors_str = "Unknown"

    # Try Google Books API first
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        response = requests.get(url, timeout=2)

        if response.status_code == 200:
            data = response.json()
            if data.get("totalItems", 0) > 0:
                items = data.get("items", [])
                if items:
                    volume = items[0].get("volumeInfo", {})
                    image_links = volume.get("imageLinks", {})
                    cover_candidate = (
                        image_links.get("extraLarge") or
                        image_links.get("large") or
                        image_links.get("medium") or
                        image_links.get("small") or
                        image_links.get("thumbnail")
                    )
                    if cover_candidate:
                        cover = cover_candidate.replace("http://", "https://")

                    authors = volume.get("authors") or []
                    if authors:
                        authors_str = ", ".join(authors)
    except Exception:
        pass  # Fall through to Open Library

    # Try Open Library as fallback for cover (author data not available there)
    if cover == PLACEHOLDER_COVER:
        try:
            url = f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
            response = requests.head(url, timeout=1)
            if response.status_code == 200:
                cover = url
        except Exception:
            pass

    return cover, authors_str


def fetch_covers_batch(books_data: list) -> list:
    """
    Fetch covers for a batch of books.
    
    Args:
        books_data: List of dicts with 'isbn' and 'title' keys
    
    Returns:
        List of dicts with added 'cover_url' key
    """
    for book in books_data:
        isbn = book.get("isbn", "")
        title = book.get("title", "")
        cover, authors = fetch_book_cover(isbn, title)
        book["thumbnail"] = cover
        if authors != "Unknown":
            book["authors"] = authors
        # Small delay to avoid rate limiting
        time.sleep(0.05)
    
    return books_data
