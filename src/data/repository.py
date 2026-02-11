"""
Unified Data Repository for book recommendation system.

Centralizes all core data access: books metadata, user history, etc.
Replaces scattered pandas.read_csv and pickle.load calls across services.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite3

from src.config import DATA_DIR
from src.core.metadata_store import metadata_store
from src.utils import setup_logger

logger = setup_logger(__name__)

# Core data file paths
BOOKS_DB_PATH = DATA_DIR / "books.db"
BOOKS_PROCESSED_CSV = DATA_DIR / "books_processed.csv"
RECALL_MODELS_DB = DATA_DIR / "recall_models.db"


class DataRepository:
    """
    Singleton data access layer. Manages loading of books_processed.csv,
    books.db, recall_models.db (user_history), etc.
    """

    _instance: Optional["DataRepository"] = None

    def __new__(cls) -> "DataRepository":
        if cls._instance is None:
            cls._instance = super(DataRepository, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._recall_conn: Optional[sqlite3.Connection] = None
        logger.info("DataRepository: Initialized (singleton)")

    def _get_recall_connection(self) -> Optional[sqlite3.Connection]:
        """Lazy SQLite connection for recall_models.db."""
        if self._recall_conn is None:
            if not RECALL_MODELS_DB.exists():
                logger.warning(f"recall_models.db not found at {RECALL_MODELS_DB}")
                return None
            try:
                self._recall_conn = sqlite3.connect(
                    str(RECALL_MODELS_DB), check_same_thread=False
                )
            except sqlite3.Error as e:
                logger.error(f"DataRepository: Failed to connect to recall DB: {e}")
        return self._recall_conn

    def get_book_metadata(self, isbn: str) -> Optional[Dict[str, Any]]:
        """
        Get book metadata by ISBN.

        Uses MetadataStore (books.db) as primary source. Returns None if not found.
        """
        meta = metadata_store.get_book_metadata(str(isbn))
        return meta if meta else None

    def get_user_history(self, user_id: str) -> List[str]:
        """
        Get user's interaction history (ISBNs) from recall_models.db.

        Used by recommendation algorithms (ItemCF, etc.). Returns empty list if
        DB unavailable or user has no history.
        """
        conn = self._get_recall_connection()
        if not conn:
            return []
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT isbn FROM user_history WHERE user_id = ?", (user_id,)
            )
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"DataRepository: get_user_history failed: {e}")
            return []

    def get_all_categories(self) -> List[str]:
        """Get unique book categories. Delegates to MetadataStore."""
        return metadata_store.get_all_categories()


# Global singleton instance
data_repository = DataRepository()
