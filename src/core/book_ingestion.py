"""
Book ingestion: persist new books to staging store (online_books.db) and ChromaDB.
Single responsibility: write path for web-discovered books; decouples from recommender.
"""
from typing import Any, Dict, Optional

from src.data.stores.metadata_store import metadata_store
from src.data.stores.online_books_store import online_books_store
from src.utils import setup_logger

logger = setup_logger(__name__)


class BookIngestion:
    """
    Persist new books to staging store + ChromaDB.
    Strategy: Staging write — no main books.db write. Decouples training data from runtime.
    """

    def __init__(self, vector_db=None, metadata_store_inst=None):
        """
        Args:
            vector_db: VectorDB instance for dense index. Lazy import to avoid circular deps.
            metadata_store_inst: For book_exists check. Defaults to global if None.
        """
        self._vector_db = vector_db
        self._meta = metadata_store_inst if metadata_store_inst is not None else metadata_store

    def _get_vector_db(self):
        if self._vector_db is None:
            from src.core.rag.vector_db import VectorDB
            self._vector_db = VectorDB()
        return self._vector_db

    def add_book(
        self,
        isbn: str,
        title: str,
        author: str,
        description: str,
        category: str = "General",
        thumbnail: Optional[str] = None,
        published_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add a new book to the staging store (online_books.db + ChromaDB).

        Args:
            isbn: ISBN-13 or ISBN-10
            title: Book title
            author: Author name(s)
            description: Book description
            category: Book category
            thumbnail: Cover image URL
            published_date: Publication date

        Returns:
            New book row dict if successful, None otherwise
        """
        try:
            isbn_s = str(isbn).strip()

            if self._meta.book_exists(isbn_s):
                logger.debug(f"Book {isbn} already exists. Skipping add.")
                return None

            new_row = {
                "isbn13": isbn_s,
                "title": title,
                "authors": author,
                "description": description,
                "simple_categories": category,
                "thumbnail": thumbnail if thumbnail else "/assets/cover-not-found.jpg",
                "average_rating": 0.0,
                "joy": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, "surprise": 0.0,
                "tags": "", "review_highlights": "",
                "isbn10": isbn_s[:10] if len(isbn_s) >= 10 else isbn_s,
                "publishedDate": published_date or "",
                "source": "google_books",
            }
            new_row["large_thumbnail"] = new_row["thumbnail"]
            new_row["image"] = new_row["thumbnail"]

            if not online_books_store.insert_book_with_fts(new_row):
                return None

            self._get_vector_db().add_book(new_row)

            logger.info(f"Successfully added book {isbn} to staging store: {title}")
            return new_row

        except Exception as e:
            logger.error(f"Error adding new book: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
