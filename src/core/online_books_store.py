"""
Online Books Store - Staging storage for freshness_fallback books.

Design: Separate SQLite file (online_books.db) decouples:
1. Data risk: Training data (books_processed.csv) stays frozen; no pollution.
2. Performance: Writes go to online_books.db only; main books.db stays read-only.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.config import DATA_DIR
from src.utils import setup_logger

logger = setup_logger(__name__)


class OnlineBooksStore:
    """
    Append-only store for books discovered via Web Search (freshness_fallback).
    Uses a separate SQLite file to avoid lock contention with main books.db.
    """

    _instance: Optional["OnlineBooksStore"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OnlineBooksStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.db_path = DATA_DIR / "online_books.db"
        self._conn = None
        self._initialized = True
        self._ensure_schema()
        logger.info("OnlineBooksStore: Initialized (staging store for web-discovered books)")

    def _ensure_schema(self) -> None:
        """Create table and FTS5 index if not exist."""
        conn = self._get_connection()
        if not conn:
            return
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS online_books (
                    isbn13 TEXT PRIMARY KEY,
                    isbn10 TEXT,
                    title TEXT,
                    authors TEXT,
                    description TEXT,
                    simple_categories TEXT,
                    thumbnail TEXT,
                    image TEXT,
                    average_rating REAL DEFAULT 0,
                    joy REAL DEFAULT 0, sadness REAL DEFAULT 0, fear REAL DEFAULT 0,
                    anger REAL DEFAULT 0, surprise REAL DEFAULT 0,
                    tags TEXT, review_highlights TEXT,
                    publishedDate TEXT,
                    source TEXT DEFAULT 'google_books'
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_online_isbn10 ON online_books (isbn10)")
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='online_books_fts'"
            )
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE VIRTUAL TABLE online_books_fts USING fts5(
                        isbn13 UNINDEXED,
                        title,
                        description,
                        authors,
                        simple_categories,
                        tokenize='porter unicode61'
                    )
                """)
            conn.commit()
        except Exception as e:
            logger.error(f"OnlineBooksStore schema setup failed: {e}")

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """Lazy connection to online_books.db (separate from main books.db)."""
        if self._conn is None:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            except Exception as e:
                logger.error(f"OnlineBooksStore: Failed to connect: {e}")
        return self._conn

    def get_book_metadata(self, isbn: str) -> Dict[str, Any]:
        """Lookup book by ISBN. Returns empty dict if not found."""
        isbn = str(isbn).strip().replace(".0", "")
        conn = self._get_connection()
        if not conn:
            return {}
        try:
            row = conn.execute(
                "SELECT * FROM online_books WHERE isbn13 = ? OR isbn10 = ?",
                (isbn, isbn),
            ).fetchone()
            return dict(row) if row else {}
        except Exception as e:
            logger.error(f"OnlineBooksStore get_book_metadata failed: {e}")
            return {}

    def book_exists(self, isbn: str) -> bool:
        """Check if ISBN exists in online store."""
        isbn = str(isbn).strip().replace(".0", "")
        conn = self._get_connection()
        if not conn:
            return False
        try:
            row = conn.execute(
                "SELECT 1 FROM online_books WHERE isbn13 = ? OR isbn10 = ? LIMIT 1",
                (isbn, isbn),
            ).fetchone()
            return row is not None
        except Exception as e:
            logger.error(f"OnlineBooksStore book_exists failed: {e}")
            return False

    def insert_book_with_fts(self, row: Dict[str, Any]) -> bool:
        """
        Insert book into online_books + FTS5. Write-only path; no lock on main DB.
        """
        conn = self._get_connection()
        if not conn:
            return False
        try:
            isbn13 = str(row.get("isbn13", ""))
            isbn10 = row.get("isbn10", isbn13[:10] if len(isbn13) >= 10 else isbn13)
            title = str(row.get("title", ""))
            authors = str(row.get("authors", ""))
            description = str(row.get("description", ""))
            categories = str(row.get("simple_categories", "General"))
            thumbnail = str(row.get("thumbnail", ""))
            image = str(row.get("image", thumbnail))
            published_date = str(row.get("publishedDate", ""))

            conn.execute(
                """
                INSERT OR IGNORE INTO online_books (
                    isbn13, isbn10, title, authors, description, simple_categories,
                    thumbnail, image, publishedDate, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'google_books')
                """,
                (isbn13, isbn10, title, authors, description, categories, thumbnail, image, published_date),
            )

            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='online_books_fts'"
            )
            if cursor.fetchone():
                cursor.execute(
                    """
                    INSERT INTO online_books_fts (isbn13, title, description, authors, simple_categories)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (isbn13, title, description, authors, categories),
                )
            conn.commit()
            logger.info(f"OnlineBooksStore: Inserted {isbn13} (staging)")
            return True
        except Exception as e:
            logger.error(f"OnlineBooksStore insert failed: {e}")
            return False

    def get_all_categories(self) -> List[str]:
        """Get unique categories from online books."""
        conn = self._get_connection()
        if not conn:
            return []
        try:
            rows = conn.execute(
                "SELECT DISTINCT simple_categories FROM online_books WHERE simple_categories != ''"
            ).fetchall()
            return [row[0] for row in rows if row[0]]
        except Exception as e:
            logger.debug(f"OnlineBooksStore get_all_categories failed: {e}")
            return []

    def fts_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        FTS5 keyword search over online_books. Used by VectorDB to merge with main FTS.
        Returns list of dicts with isbn13, title, description, authors, simple_categories.
        """
        conn = self._get_connection()
        if not conn:
            return []
        try:
            clean_query = query.strip().replace('"', '""')
            if not clean_query:
                return []
            fts_query = f'"{clean_query}"'
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT isbn13, title, description, authors, simple_categories
                FROM online_books_fts
                WHERE online_books_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, k),
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"OnlineBooksStore FTS search failed: {e}")
            return []


online_books_store = OnlineBooksStore()
