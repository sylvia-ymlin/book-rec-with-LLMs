import pytest
from unittest.mock import patch, MagicMock
from src.vector_db import VectorDB


class TestVectorDB:
    """VectorDB uses Chroma + HuggingFaceEmbeddings. No TextLoader (init_db handles indexing)."""

    @pytest.fixture
    def mock_initialization(self):
        """Mock HuggingFaceEmbeddings and Chroma to avoid loading real models."""
        with patch("src.vector_db.HuggingFaceEmbeddings") as mock_emb, \
             patch("src.vector_db.Chroma") as mock_chroma, \
             patch("src.vector_db.CHROMA_DB_DIR") as mock_dir:
            mock_dir.exists.return_value = False
            mock_dir.iterdir.return_value = []
            yield {"embeddings": mock_emb, "chroma": mock_chroma, "dir": mock_dir}

    def test_singleton_pattern(self, mock_initialization):
        """Ensure VectorDB is a singleton."""
        VectorDB._instance = None
        db1 = VectorDB()
        db2 = VectorDB()
        assert db1 is db2

    def test_search(self, mock_initialization):
        """Test search delegation to ChromaDB."""
        VectorDB._instance = None
        db = VectorDB()
        if db.db:
            db.db.similarity_search.return_value = ["doc1", "doc2"]
            results = db.search("test query", k=5)
            assert len(results) == 2
            db.db.similarity_search.assert_called_with("test query", k=5)
        else:
            results = db.search("test query", k=5)
            assert results == []

    def test_initialization_loads_existing_db(self, mock_initialization):
        """When DB dir exists, Chroma loads from persist_directory."""
        VectorDB._instance = None
        mock_initialization["dir"].exists.return_value = True
        mock_initialization["dir"].iterdir.return_value = [1]
        VectorDB()
        mock_initialization["chroma"].assert_called_once()

    def test_sparse_fts_search(self, mock_initialization):
        """Test FTS5 keyword search."""
        VectorDB._instance = None
        db = VectorDB()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        row = {
            "isbn13": "123",
            "title": "Test Book",
            "description": "Desc",
            "authors": "Auth",
            "simple_categories": "Cat",
        }
        mock_cursor.fetchall.return_value = [row]
        with patch("src.vector_db.metadata_store") as mock_store, \
             patch("src.vector_db.online_books_store") as mock_online:
            mock_store.connection = mock_conn
            mock_online.fts_search.return_value = []
            db.fts_enabled = True
            results = db._sparse_fts_search("query")
            assert len(results) == 1
            assert results[0].metadata["isbn"] == "123"

    def test_hybrid_search_fts5(self, mock_initialization):
        """Test Hybrid Search with FTS5."""
        VectorDB._instance = None
        db = VectorDB()
        
        with patch.object(db, 'search', return_value=[]), \
             patch.object(db, '_sparse_fts_search', return_value=[]):
            db.fts_enabled = True
            db.db = MagicMock()
            
            results = db.hybrid_search("query")
            assert results == []
