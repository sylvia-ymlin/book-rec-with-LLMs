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
import csv
import ast

# Placeholder image for books without covers (local asset)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLACEHOLDER_COVER = str(PROJECT_ROOT / "assets" / "cover-not-found.jpg")
_LOCAL_META_INDEX: dict[str, tuple[str, str]] | None = None


def _normalize_cover_url(url: str) -> str:
    return str(url or "").strip().replace("http://", "https://")


def _normalize_authors(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "Unknown"
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                names = [str(x).strip() for x in parsed if str(x).strip()]
                return ", ".join(names) if names else "Unknown"
        except Exception:
            pass
    if ";" in text:
        names = [x.strip() for x in text.split(";") if x.strip()]
        return ", ".join(names) if names else "Unknown"
    return text


def _build_local_meta_index() -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}

    # Primary: clean local data with thumbnail + authors
    candidates = [
        PROJECT_ROOT / "data" / "books_with_emotions.csv",
        PROJECT_ROOT / "data" / "books_basic_info.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    isbn13 = str(row.get("isbn13", "")).strip()
                    isbn10 = str(row.get("isbn10", "")).strip()
                    cover = _normalize_cover_url(row.get("thumbnail") or row.get("image") or "")
                    authors = _normalize_authors(row.get("authors", ""))
                    if not cover:
                        continue
                    if isbn13:
                        index.setdefault(isbn13, (cover, authors))
                    if isbn10:
                        index.setdefault(isbn10, (cover, authors))
        except Exception:
            # If one file is malformed, continue with others.
            continue
    return index


def _get_local_meta(isbn: str) -> tuple[str, str]:
    global _LOCAL_META_INDEX
    if _LOCAL_META_INDEX is None:
        _LOCAL_META_INDEX = _build_local_meta_index()
    return _LOCAL_META_INDEX.get(str(isbn).strip(), ("", "Unknown"))


def _pick_google_cover(volume: dict) -> str:
    image_links = volume.get("imageLinks", {}) or {}
    candidate = (
        image_links.get("extraLarge")
        or image_links.get("large")
        or image_links.get("medium")
        or image_links.get("small")
        or image_links.get("thumbnail")
        or ""
    )
    if candidate:
        return candidate.replace("http://", "https://")
    return ""


def _query_google_books(query: str, timeout: float = 1.2) -> dict:
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        return {}
    data = response.json()
    items = data.get("items", [])
    if not items:
        return {}
    return items[0].get("volumeInfo", {}) or {}


def _openlibrary_cover_url(isbn: str) -> str:
    # default=false: return 404 when no cover exists (avoid fake tiny placeholder)
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false"


def _openlibrary_lookup(isbn: str, timeout: float = 1.0) -> dict:
    url = (
        "https://openlibrary.org/api/books"
        f"?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    )
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        return {}
    payload = response.json() or {}
    return payload.get(f"ISBN:{isbn}", {}) or {}


@lru_cache(maxsize=1000)
def fetch_book_cover(isbn: str, title: str = "") -> tuple[str, str, str]:
    """
    Fetch book cover URL (Google Books -> Open Library), authors and description.

    Returns:
        (cover_url, authors_str, description_from_api)
    """
    cover = PLACEHOLDER_COVER
    authors_str = "Unknown"
    api_description = ""

    isbn = str(isbn or "").strip()
    title = str(title or "").strip()

    # Local CSV enrichment first: much faster and more stable than network calls.
    local_cover, local_authors = _get_local_meta(isbn)
    if local_cover:
        cover = local_cover
    if local_authors != "Unknown":
        authors_str = local_authors

    # Try Google Books API by ISBN first
    try:
        volume = _query_google_books(f"isbn:{isbn}")
        if volume:
            cover_candidate = _pick_google_cover(volume)
            if cover_candidate:
                cover = cover_candidate

            authors = volume.get("authors") or []
            if authors:
                authors_str = ", ".join(authors)
            api_description = volume.get("description") or api_description
    except Exception:
        pass

    # If still incomplete, fallback to title search in Google Books
    if (cover == PLACEHOLDER_COVER or authors_str == "Unknown") and title:
        try:
            volume = _query_google_books(f"intitle:{title}")
            if volume:
                cover_candidate = _pick_google_cover(volume)
                if cover == PLACEHOLDER_COVER and cover_candidate:
                    cover = cover_candidate

                authors = volume.get("authors") or []
                if authors_str == "Unknown" and authors:
                    authors_str = ", ".join(authors)
                if not api_description:
                    api_description = volume.get("description") or api_description
        except Exception:
            pass

    # OpenLibrary data API fallback for authors/description/cover metadata
    if isbn and (cover == PLACEHOLDER_COVER or authors_str == "Unknown" or not api_description):
        try:
            ol = _openlibrary_lookup(isbn)
            if ol:
                if authors_str == "Unknown":
                    ol_authors = [a.get("name", "").strip() for a in ol.get("authors", []) if a.get("name")]
                    if ol_authors:
                        authors_str = ", ".join(ol_authors)
                if not api_description:
                    ol_desc = ol.get("description", "")
                    if isinstance(ol_desc, dict):
                        api_description = str(ol_desc.get("value", "")).strip()
                    elif isinstance(ol_desc, str):
                        api_description = ol_desc.strip()
                if cover == PLACEHOLDER_COVER:
                    ol_covers = ol.get("cover", {}) or {}
                    cover = (
                        ol_covers.get("large")
                        or ol_covers.get("medium")
                        or ol_covers.get("small")
                        or cover
                    )
                    if cover:
                        cover = _normalize_cover_url(cover)
        except Exception:
            pass

    # Try Open Library cover endpoint as final fallback
    if cover == PLACEHOLDER_COVER:
        try:
            url = _openlibrary_cover_url(isbn)
            response = requests.get(url, timeout=0.8)
            if response.status_code == 200:
                cover = url
        except Exception:
            pass

    return cover, authors_str, api_description


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
        cover, authors, api_desc = fetch_book_cover(isbn, title)
        book["thumbnail"] = cover
        if authors != "Unknown":
            book["authors"] = authors
        if api_desc:
            book["description_api"] = api_desc
        # Small delay to avoid rate limiting
        time.sleep(0.05)

    return books_data

