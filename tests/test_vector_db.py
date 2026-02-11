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
        # Mock the underlying Chroma instance's similarity_search
        db.db.similarity_search.return_value = ["doc1", "doc2"]
        
        results = db.search("test query", k=5)
        
        assert len(results) == 2
        db.db.similarity_search.assert_called_with("test query", k=5)
    
    def test_initialization_creates_new_db(self, mock_initialization):
        """Test logic when DB directory does not exist."""
        VectorDB._instance = None
        
        mock_initialization['dir'].exists.return_value = False
        
        VectorDB()
        
        # Should initiate TextLoader and Chroma.from_documents
        mock_initialization['loader'].assert_called()
        mock_initialization['chroma'].from_documents.assert_called()

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
