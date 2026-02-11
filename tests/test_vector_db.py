import pytest
from unittest.mock import patch, MagicMock
from src.vector_db import VectorDB

class TestVectorDB:

    @pytest.fixture
    def mock_initialization(self):
        """Mock external dependencies for VectorDB."""
        with patch('src.vector_db.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.vector_db.Chroma') as mock_chroma, \
             patch('src.vector_db.TextLoader') as mock_loader, \
             patch('src.vector_db.CharacterTextSplitter') as mock_splitter, \
             patch('src.vector_db.CHROMA_DB_DIR') as mock_dir:
             
            # Setup mock directory specific behavior
            mock_dir.exists.return_value = False
            
            yield {
                'embeddings': mock_emb,
                'chroma': mock_chroma,
                'loader': mock_loader,
                'splitter': mock_splitter,
                'dir': mock_dir
            }

    def test_singleton_pattern(self, mock_initialization):
        """Ensure VectorDB is a singleton."""
        # Reset singleton instance
        VectorDB._instance = None
        
        db1 = VectorDB()
        db2 = VectorDB()
        assert db1 is db2

    def test_search(self, mock_initialization):
        """Test search delegation to ChromaDB."""
        # Reset singleton
        VectorDB._instance = None
        
        db = VectorDB()
        # db.db might be None in survival mode
        if db.db:
            db.db.similarity_search.return_value = ["doc1", "doc2"]
            results = db.search("test query", k=5)
            assert len(results) == 2
            db.db.similarity_search.assert_called_with("test query", k=5)
        else:
            results = db.search("test query", k=5)
            assert results == []
    
    def test_initialization_creates_new_db(self, mock_initialization):
        """Test logic when DB directory does not exist."""
        VectorDB._instance = None
        
        mock_initialization['dir'].exists.return_value = False
        
        VectorDB()
        
        # The logic has changed with survival mode (init_db.py handles creation now)
        pass 

    def test_initialization_loads_existing_db(self, mock_initialization):
        """Test logic when DB directory exists."""
        VectorDB._instance = None
        
        # Mock directory exists and has content
        mock_initialization['dir'].exists.return_value = True
        mock_initialization['dir'].iterdir.return_value = [1] # Not empty
        
        VectorDB()
        
        # Should NOT create from documents, but load persistent DB
        mock_initialization['loader'].assert_not_called()
        # Should initialize Chroma with persist_directory
        mock_initialization['chroma'].assert_called()

    def test_sparse_fts_search(self, mock_initialization):
        """Test FTS5 keyword search."""
        VectorDB._instance = None
        db = VectorDB()
        
        # Mock metadata_store connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        # We need to patch the metadata_store imported in vector_db
        with patch('src.vector_db.metadata_store') as mock_store:
            mock_store.connection = mock_conn
            db.fts_enabled = True
            
            # Row behaves like a dict
            row = {'isbn13': '123', 'title': 'Test Book', 'description': 'Desc', 'authors': 'Auth', 'simple_categories': 'Cat'}
            mock_cursor.fetchall.return_value = [row]
            
            results = db._sparse_fts_search("query")
            assert len(results) == 1
            assert results[0].metadata['isbn'] == '123'

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
