"""
Fallback provider for RAG: fetch books from external sources (e.g. Google Books API)
when local results are insufficient.
"""
from typing import Any, Dict, List
import sqlite3

from src.data.stores.metadata_store import metadata_store
from src.core.response_formatter import format_web_book_response
from src.infra.utils import setup_logger


logger = setup_logger(__name__)


class FallbackProvider:
    """
    Fetch books from Google Books API when local search is insufficient.
    Persists discovered books via BookIngestion for future queries.
    """

    def __init__(self, book_ingestion=None, metadata_store_inst=None):
        """
        Args:
            book_ingestion: BookIngestion instance for persisting. Lazy init if None.
            metadata_store_inst: For book_exists check. Defaults to global if None.
        """
        from src.core.book_ingestion import BookIngestion

        self._meta = (
            metadata_store_inst if metadata_store_inst is not None else metadata_store
        )
        self._ingestion = book_ingestion or BookIngestion(metadata_store_inst=self._meta)

    async def fetch_async(
        self,
        query: str,
        max_results: int,
        category: str = "All",
    ) -> List[Dict[str, Any]]:
        """
        Async: Fetch books from Google Books API.
        Uses httpx to avoid blocking the FastAPI event loop.
        """
        try:
            from src.rag.web_search import search_google_books_async
        except ImportError:
            logger.warning("Web search module not available")
            return []

        results: List[Dict[str, Any]] = []
        try:
            web_books = await search_google_books_async(query, max_results=max_results * 2)

            for book in web_books:
                isbn = book.get("isbn13", "")
                if not isbn:
                    continue
                if self._meta.book_exists(isbn):
                    continue
                if category and category != "All":
                    book_cat = book.get("simple_categories", "")
                    if category.lower() not in (book_cat or "").lower():
                        continue

                added = self._ingestion.add_book(
                    isbn=isbn,
                    title=book.get("title", ""),
                    author=book.get("authors", "Unknown"),
                    description=book.get("description", ""),
                    category=book.get("simple_categories", "General"),
                    thumbnail=book.get("thumbnail"),
                    published_date=book.get("publishedDate", ""),
                )
                if added:
                    results.append(format_web_book_response(book, isbn))
                if len(results) >= max_results:
                    break

            logger.info(
                "Web fallback: Found and persisted %d new books",
                len(results),
            )
            return results
        except sqlite3.Error as e:
            logger.error("[WebFallback:DB_ERROR] query='%s' - %s", query, e)
            return []
        except Exception as e:
            logger.exception(
                "[WebFallback:UNEXPECTED] query='%s' - %s: %s",
                query,
                type(e).__name__,
                e,
            )
            return []

    def fetch_sync(
        self,
        query: str,
        max_results: int,
        category: str = "All",
    ) -> List[Dict[str, Any]]:
        """
        Sync: Fetch books from Google Books API.
        For scripts/CLI; prefer fetch_async in FastAPI.
        """
        try:
            from src.rag.web_search import search_google_books
        except ImportError:
            logger.warning("Web search module not available")
            return []

        results: List[Dict[str, Any]] = []
        try:
            web_books = search_google_books(query, max_results=max_results * 2)

            for book in web_books:
                isbn = book.get("isbn13", "")
                if not isbn:
                    continue
                if self._meta.book_exists(isbn):
                    continue
                if category and category != "All":
                    book_cat = book.get("simple_categories", "")
                    if category.lower() not in (book_cat or "").lower():
                        continue

                added = self._ingestion.add_book(
                    isbn=isbn,
                    title=book.get("title", ""),
                    author=book.get("authors", "Unknown"),
                    description=book.get("description", ""),
                    category=book.get("simple_categories", "General"),
                    thumbnail=book.get("thumbnail"),
                    published_date=book.get("publishedDate", ""),
                )
                if added:
                    results.append(format_web_book_response(book, isbn))
                if len(results) >= max_results:
                    break

            logger.info(
                "Web fallback: Found and persisted %d new books",
                len(results),
            )
            return results
        except sqlite3.Error as e:
            logger.error("[WebFallback:DB_ERROR] query='%s' - %s", query, e)
            return []
        except Exception as e:
            logger.exception(
                "[WebFallback:UNEXPECTED] query='%s' - %s: %s",
                query,
                type(e).__name__,
                e,
            )
            return []


__all__ = ["FallbackProvider"]

