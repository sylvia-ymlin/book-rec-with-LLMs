import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.config import DATA_DIR

logger = logging.getLogger(__name__)

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
        """Fast lookup for book metadata by ISBN (10 or 13) using SQLite index."""
        isbn = str(isbn).strip().replace(".0", "")
        row = self._query_one("SELECT * FROM books WHERE isbn13 = ? OR isbn10 = ?", (isbn, isbn))
        return dict(row) if row else {}

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
        """Efficiently fetch unique categories from SQLite."""
        conn = self.connection
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT simple_categories FROM books")
            return [row[0] for row in cursor.fetchall() if row[0]]
        return []

    def load_books_processed(self): pass
    def load_train_data(self): pass

# Global access point
metadata_store = MetadataStore()
