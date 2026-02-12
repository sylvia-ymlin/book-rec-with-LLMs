"""
Web Search Fallback Module for Data Freshness (RAG path).

Extends the existing Google Books integration to support
full book metadata retrieval and keyword-based search for new books.
"""
from functools import lru_cache
from datetime import datetime
from typing import Optional

import requests

from src.utils import setup_logger


logger = setup_logger(__name__)

# Google Books API endpoint
GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"

# Request timeout (seconds)
REQUEST_TIMEOUT = 5


def _parse_isbn_from_identifiers(identifiers: list[dict]) -> tuple[str, str]:
    """
    Extract ISBN-13 and ISBN-10 from Google Books industryIdentifiers.

    Returns:
        (isbn13, isbn10) - empty strings if not found
    """
    isbn13, isbn10 = "", ""
    for ident in identifiers:
        id_type = ident.get("type", "")
        id_value = ident.get("identifier", "")
        if id_type == "ISBN_13":
            isbn13 = id_value
        elif id_type == "ISBN_10":
            isbn10 = id_value
    return isbn13, isbn10


def _parse_volume_info(volume_info: dict) -> Optional[dict]:
    """
    Parse Google Books volumeInfo into our standard book format.

    Returns:
        dict with keys: isbn13, isbn10, title, authors, description,
                       publishedDate, thumbnail, categories
        None if ISBN is missing (we can't index without ISBN)
    """
    identifiers = volume_info.get("industryIdentifiers", [])
    isbn13, isbn10 = _parse_isbn_from_identifiers(identifiers)

    # Skip books without ISBN - can't be indexed reliably
    if not isbn13 and not isbn10:
        return None

    # Extract image links (prefer larger sizes)
    image_links = volume_info.get("imageLinks", {})
    thumbnail = (
        image_links.get("extraLarge")
        or image_links.get("large")
        or image_links.get("medium")
        or image_links.get("small")
        or image_links.get("thumbnail")
        or ""
    )
    # Ensure HTTPS
    if thumbnail.startswith("http://"):
        thumbnail = thumbnail.replace("http://", "https://")

    # Categories: Google returns list, we join to single string
    categories = volume_info.get("categories", [])
    category_str = categories[0] if categories else "General"

    return {
        "isbn13": isbn13 or isbn10,  # Primary key
        "isbn10": isbn10 or (isbn13[:10] if isbn13 else ""),
        "title": volume_info.get("title", "Unknown Title"),
        "authors": ", ".join(volume_info.get("authors", ["Unknown"])),
        "description": volume_info.get("description", ""),
        "publishedDate": volume_info.get("publishedDate", ""),
        "thumbnail": thumbnail,
        "simple_categories": category_str,
        "average_rating": volume_info.get("averageRating", 0.0),
        "source": "google_books",  # Track data source
    }


def _log_google_books_error(kind: str, query: str, detail: str = "") -> None:
    """Log with [GoogleBooks:KIND] prefix for monitoring/grep."""
    msg = f"[GoogleBooks:{kind}] query='{query}'"
    if detail:
        msg += f" - {detail}"
    if kind == "RATE_LIMIT":
        logger.error(msg)  # 429 needs alerting
    elif kind in ("TIMEOUT", "NETWORK", "SERVER_ERROR"):
        logger.warning(msg)
    else:
        logger.warning(msg)


def search_google_books(query: str, max_results: int = 10) -> list[dict]:
    """
    Search Google Books by keyword query.
    """
    if not query or not query.strip():
        return []

    max_results = min(max_results, 40)  # Google's API limit

    try:
        params = {
            "q": query,
            "maxResults": max_results,
            "printType": "books",
            "orderBy": "relevance",
        }

        response = requests.get(
            GOOGLE_BOOKS_API,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429:
            _log_google_books_error("RATE_LIMIT", query, "quota exceeded (429)")
            return []
        if response.status_code >= 500:
            _log_google_books_error(
                "SERVER_ERROR", query, f"status={response.status_code}"
            )
            return []
        if response.status_code != 200:
            _log_google_books_error(
                "HTTP_ERROR", query, f"status={response.status_code}"
            )
            return []

        data = response.json()
        total_items = data.get("totalItems", 0)

        if total_items == 0:
            logger.info("No results for query: %s", query)
            return []

        items = data.get("items", [])
        results: list[dict] = []

        for item in items:
            volume_info = item.get("volumeInfo", {})
            parsed = _parse_volume_info(volume_info)
            if parsed:
                results.append(parsed)

        logger.info(
            "Google Books search '%s': %d valid results", query, len(results)
        )
        return results

    except requests.Timeout:
        _log_google_books_error("TIMEOUT", query)
        return []
    except requests.ConnectionError as e:
        _log_google_books_error("NETWORK", query, str(e))
        return []
    except requests.RequestException as e:
        _log_google_books_error("REQUEST_ERROR", query, str(e))
        return []
    except Exception as e:
        logger.exception("[GoogleBooks:UNEXPECTED] query='%s' - %s", query, e)
        return []


async def search_google_books_async(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    """
    Async version: Search Google Books by keyword query.
    Uses httpx to avoid blocking the event loop in FastAPI.
    """
    if not query or not query.strip():
        return []

    max_results = min(max_results, 40)

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available, falling back to sync")
        return search_google_books(query, max_results)

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(
                GOOGLE_BOOKS_API,
                params={
                    "q": query,
                    "maxResults": max_results,
                    "printType": "books",
                    "orderBy": "relevance",
                },
            )
    except httpx.TimeoutException:
        _log_google_books_error("TIMEOUT", query)
        return []
    except httpx.ConnectError as e:
        _log_google_books_error("NETWORK", query, str(e))
        return []
    except httpx.HTTPError as e:
        _log_google_books_error("REQUEST_ERROR", query, str(e))
        return []

    if response.status_code == 429:
        _log_google_books_error("RATE_LIMIT", query, "quota exceeded (429)")
        return []
    if response.status_code >= 500:
        _log_google_books_error(
            "SERVER_ERROR", query, f"status={response.status_code}"
        )
        return []
    if response.status_code != 200:
        _log_google_books_error(
            "HTTP_ERROR", query, f"status={response.status_code}"
        )
        return []

    try:
        data = response.json()
    except Exception as e:
        logger.warning("[GoogleBooks:PARSE_ERROR] query='%s' - %s", query, e)
        return []

    total_items = data.get("totalItems", 0)
    if total_items == 0:
        logger.info("No results for query: %s", query)
        return []

    items = data.get("items", [])
    results: list[dict] = []
    for item in items:
        volume_info = item.get("volumeInfo", {})
        parsed = _parse_volume_info(volume_info)
        if parsed:
            results.append(parsed)

    logger.info(
        "Google Books search '%s': %d valid results", query, len(results)
    )
    return results


@lru_cache(maxsize=500)
def fetch_book_by_isbn(isbn: str) -> Optional[dict]:
    """
    Fetch complete book metadata by ISBN from Google Books.
    """
    if not isbn or not isbn.strip():
        return None

    isbn = isbn.strip().replace("-", "")

    try:
        params = {
            "q": f"isbn:{isbn}",
            "maxResults": 1,
        }

        response = requests.get(
            GOOGLE_BOOKS_API,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429:
            _log_google_books_error(
                "RATE_LIMIT", f"isbn:{isbn}", "quota exceeded (429)"
            )
            return None
        if response.status_code != 200:
            return None

        data = response.json()
        if data.get("totalItems", 0) == 0:
            return None

        items = data.get("items", [])
        if not items:
            return None

        volume_info = items[0].get("volumeInfo", {})
        return _parse_volume_info(volume_info)

    except requests.Timeout:
        _log_google_books_error("TIMEOUT", f"isbn:{isbn}")
        return None
    except requests.ConnectionError as e:
        _log_google_books_error("NETWORK", f"isbn:{isbn}", str(e))
        return None
    except requests.RequestException as e:
        logger.debug("fetch_book_by_isbn(%s) failed: %s", isbn, e)
        return None
    except Exception as e:
        logger.exception(
            "[GoogleBooks:UNEXPECTED] fetch_book_by_isbn(%s) - %s", isbn, e
        )
        return None


def search_new_books_by_category(
    category: str,
    year: Optional[int] = None,
    max_results: int = 10,
) -> list[dict]:
    """
    Search for recently published books in a specific category.
    """
    if year is None:
        year = datetime.now().year

    # Build query with subject filter
    query = f"subject:{category}"

    # Get more results than needed, filter by year locally
    raw_results = search_google_books(query, max_results=max_results * 2)

    filtered = []
    for book in raw_results:
        pub_date = book.get("publishedDate", "")
        if pub_date:
            # Extract year from various date formats (YYYY, YYYY-MM, YYYY-MM-DD)
            try:
                pub_year = int(pub_date[:4])
                if pub_year >= year:
                    filtered.append(book)
            except (ValueError, IndexError):
                continue

    return filtered[:max_results]


def is_fresh_enough(
    results: list[dict],
    threshold: int = 3,
    min_year: Optional[int] = None,
) -> bool:
    """
    Evaluate if local search results meet freshness requirements.
    """
    if len(results) < threshold:
        return False

    if min_year is None:
        return True

    # Count books meeting year requirement
    fresh_count = 0
    for book in results:
        pub_date = book.get("publishedDate", "") or book.get(
            "published_date", ""
        )
        if pub_date:
            try:
                pub_year = int(str(pub_date)[:4])
                if pub_year >= min_year:
                    fresh_count += 1
            except (ValueError, IndexError):
                continue

    return fresh_count >= threshold


def extract_year_from_query(query: str) -> Optional[int]:
    """
    Extract year requirement from user query.
    """
    import re

    # Explicit year patterns
    year_patterns = [
        r"\b(202[0-9])\b",  # 2020-2029
        r"\b(201[0-9])\b",  # 2010-2019
    ]

    for pattern in year_patterns:
        match = re.search(pattern, query)
        if match:
            return int(match.group(1))

    # Keywords implying "recent" = current year - 1
    freshness_keywords = {
        "new",
        "newest",
        "latest",
        "recent",
        "modern",
        "contemporary",
        "current",
    }

    words = set(query.lower().split())
    if words & freshness_keywords:
        return datetime.now().year - 1

    return None


__all__ = [
    "search_google_books",
    "search_google_books_async",
    "fetch_book_by_isbn",
    "search_new_books_by_category",
    "is_fresh_enough",
    "extract_year_from_query",
]

