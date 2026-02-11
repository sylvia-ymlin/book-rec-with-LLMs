import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.config import DATA_DIR
from src.utils import setup_logger

logger = setup_logger(__name__)

# Lazy import to avoid circular dependency
def _online_store():
    from src.core.online_books_store import online_books_store
    return online_books_store

class MetadataStore:
    """
    Singleton class to manage large book metadata efficiently.
    
    ENGINEERING IMPROVEMENT:
    Transitioned from in-memory Pandas/Dict structures to a zero-RAM SQLite architecture.
    This allows the application to handle the full 221k book dataset within the 16Gi RAM
    limit of Hugging Face Spaces by performing on-demand indexed lookups instead of 
    pre-loading the entire corpus.
    
    Features:
    - Zero-RAM footprint for idle metadata.
    - FTS5 Virtual Tables for BM25-based keyword search (replaces rank_bm25).
    """
    _instance: Optional['MetadataStore'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetadataStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.db_path = DATA_DIR / "books.db"
        self._conn = None
        self._initialized = True
        logger.info(f"MetadataStore: Initialized (Zero-RAM mode using SQLite)")

    @property
    def connection(self):
        """Lazy-loaded SQLite connection."""
        if self._conn is None:
            if not self.db_path.exists():
                logger.warning(f"MetadataStore: {self.db_path} not found. Survival mode active.")
                return None
            try:
                self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            except Exception as e:
                logger.error(f"MetadataStore: Failed to connect to SQLite: {e}")
        return self._conn

    def _query_one(self, query: str, params: tuple) -> Optional[sqlite3.Row]:
        conn = self.connection
        if not conn: return None
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"MetadataStore Query Error: {e}")
            return None

    def get_book_metadata(self, isbn: str) -> Dict[str, Any]:
        """Fast lookup: main store first, then online staging store (read path stays fast)."""
        isbn = str(isbn).strip().replace(".0", "")
        row = self._query_one("SELECT * FROM books WHERE isbn13 = ? OR isbn10 = ?", (isbn, isbn))
        if row:
            return dict(row)
        return _online_store().get_book_metadata(isbn) or {}

    def get_image(self, isbn: str, default: str = "") -> str:
        isbn = str(isbn).strip().replace(".0", "")
        row = self._query_one("SELECT image FROM books WHERE isbn13 = ? OR isbn10 = ?", (isbn, isbn))
        return row['image'] if row and row['image'] else default

    def get_rating(self, isbn: str, default: float = 0.0) -> float:
        isbn = str(isbn).strip().replace(".0", "")
        row = self._query_one("SELECT average_rating FROM books WHERE isbn13 = ? OR isbn10 = ?", (isbn, isbn))
        try:
            return float(row['average_rating']) if row and row['average_rating'] else default
        except:
            return default

    # Legacy compatibility properties
    @property
    def isbn_to_title(self) -> Dict[str, str]:
        return {} # Should use get_book_metadata()

    @property
    def item_category(self) -> Dict[str, str]:
        return {} 

    @property
    def item_author(self) -> Dict[str, str]:
        return {}

    @property
    def user_stats(self) -> Dict[str, Any]: return {}
    @property
    def item_stats(self) -> Dict[str, Any]: return {}
    
    @property
    def books_df(self) -> pd.DataFrame:
        """
        [DEPRECATED] DANGER: This loads all 221k books into RAM.
        Use get_book_metadata() or SQL queries instead.
        """
        logger.warning("MetadataStore: Loading full books_df into RAM. This is a potential OOM risk!")
        conn = self.connection
        if conn:
            return pd.read_sql("SELECT * FROM books", conn)
        return pd.DataFrame()

    def get_all_categories(self) -> List[str]:
        """Efficiently fetch unique categories from main + online store."""
        conn = self.connection
        cats = set()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT simple_categories FROM books")
            cats.update(row[0] for row in cursor.fetchall() if row[0])
        cats.update(_online_store().get_all_categories())
        return sorted(cats)

    def insert_book(self, row: Dict[str, Any]) -> bool:
        """Insert a new book for add_new_book. Maps thumbnail->image if needed."""
        conn = self.connection
        if not conn:
            return False
        try:
            info = conn.execute("PRAGMA table_info(books)").fetchall()
            table_cols = [c[1] for c in info]
            row = dict(row)
            if "image" in table_cols and "image" not in row and "thumbnail" in row:
                row["image"] = row["thumbnail"]
            cols = [c for c in table_cols if c in row]
            vals = [row[c] for c in cols]
            ph = ",".join("?" * len(cols))
            conn.execute(f"INSERT OR IGNORE INTO books ({','.join(cols)}) VALUES ({ph})", vals)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"MetadataStore insert_book failed: {e}")
            return False

    def insert_book_with_fts(self, row: Dict[str, Any]) -> bool:
        """
        Insert a new book into both main table AND FTS5 index.
        
        This enables incremental indexing - new books are immediately searchable
        via keyword search without requiring a full index rebuild.
        
        Args:
            row: Book data dict with keys: isbn13, title, description, authors, simple_categories, etc.
        
        Returns:
            True if successful, False otherwise
        """
        conn = self.connection
        if not conn:
            return False
        
        try:
            # 1. Insert into main books table
            if not self.insert_book(row):
                return False
            
            # 2. Insert into FTS5 index
            # FTS5 columns: isbn13, title, description, authors, simple_categories
            isbn13 = str(row.get("isbn13", ""))
            title = str(row.get("title", ""))
            description = str(row.get("description", ""))
            authors = str(row.get("authors", ""))
            categories = str(row.get("simple_categories", ""))
            
            # Check if FTS5 table exists
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='books_fts'"
            )
            if not cursor.fetchone():
                logger.warning("MetadataStore: FTS5 table 'books_fts' not found. Skipping FTS index.")
                return True  # Main insert succeeded, FTS just not available
            
            # Insert into FTS5 (use INSERT OR REPLACE to handle updates)
            cursor.execute(
                """
                INSERT OR REPLACE INTO books_fts (isbn13, title, description, authors, simple_categories)
                VALUES (?, ?, ?, ?, ?)
                """,
                (isbn13, title, description, authors, categories)
            )
            conn.commit()
            
            logger.info(f"MetadataStore: Inserted book {isbn13} into FTS5 index")
            return True
            
        except sqlite3.OperationalError as e:
            # FTS5 might not support OR REPLACE, try without
            if "REPLACE" in str(e):
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO books_fts (isbn13, title, description, authors, simple_categories)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (isbn13, title, description, authors, categories)
                    )
                    conn.commit()
                    return True
                except Exception as inner_e:
                    logger.error(f"MetadataStore FTS5 insert failed: {inner_e}")
                    return False
            logger.error(f"MetadataStore FTS5 insert failed: {e}")
            return False
        except Exception as e:
            logger.error(f"MetadataStore insert_book_with_fts failed: {e}")
            return False

    def book_exists(self, isbn: str) -> bool:
        """Check if ISBN exists in main or online staging store."""
        isbn = str(isbn).strip().replace(".0", "")
        row = self._query_one(
            "SELECT 1 FROM books WHERE isbn13 = ? OR isbn10 = ? LIMIT 1",
            (isbn, isbn)
        )
        if row:
            return True
        return _online_store().book_exists(isbn)

    def get_newest_book_year(self) -> Optional[int]:
        """Get the publication year of the newest book in the database."""
        conn = self.connection
        if not conn:
            return None
        try:
            cursor = conn.cursor()
            # Try publishedDate column
            cursor.execute(
                "SELECT publishedDate FROM books WHERE publishedDate IS NOT NULL "
                "ORDER BY publishedDate DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row and row[0]:
                # Extract year from date string
                date_str = str(row[0])
                if len(date_str) >= 4:
                    return int(date_str[:4])
        except Exception as e:
            logger.debug(f"get_newest_book_year failed: {e}")
        return None

    def get_book_count(self) -> int:
        """Get total number of books in the database."""
        conn = self.connection
        if not conn:
            return 0
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM books")
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.error(f"get_book_count failed: {e}")
            return 0

    def get_books_by_year_distribution(self) -> Dict[int, int]:
        """Get distribution of books by publication year."""
        conn = self.connection
        if not conn:
            return {}
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT SUBSTR(publishedDate, 1, 4) as year, COUNT(*) as count
                FROM books
                WHERE publishedDate IS NOT NULL AND LENGTH(publishedDate) >= 4
                GROUP BY year
                ORDER BY year DESC
                LIMIT 20
                """
            )
            return {int(row[0]): row[1] for row in cursor.fetchall() if row[0].isdigit()}
        except Exception as e:
            logger.debug(f"get_books_by_year_distribution failed: {e}")
            return {}

    def load_books_processed(self): pass
    def load_train_data(self): pass

# Global access point
metadata_store = MetadataStore()
