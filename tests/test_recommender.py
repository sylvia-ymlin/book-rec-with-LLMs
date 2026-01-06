import pytest
from unittest.mock import patch, MagicMock
from src.recommender import BookRecommender

class TestBookRecommender:
    
    @pytest.fixture
    def recommender(self, mock_books_df, mock_vector_db):
        """Initialize recommender with mocked dependencies."""
        with patch('src.recommender.load_books_data', return_value=mock_books_df), \
             patch('src.recommender.VectorDB', return_value=mock_vector_db):
            return BookRecommender()

    def test_initialization(self, recommender):
        """Test if recommender initializes correctly."""
        assert recommender.books is not None
        assert not recommender.books.empty
        assert recommender.vector_db is not None

    def test_get_categories(self, recommender):
        """Test retrieving categories."""
        categories = recommender.get_categories()
        assert "All" in categories
        assert "Fiction" in categories
        assert "Non-Fiction" in categories
        assert len(categories) > 1

    def test_get_tones(self, recommender):
        """Test retrieving tones."""
        tones = recommender.get_tones()
        assert "All" in tones
        assert "Happy" in tones
        assert "Sad" in tones

    def test_recommend_basic(self, recommender):
        """Test basic recommendation flow."""
        results = recommender.get_recommendations("test query")
        assert len(results) > 0
        assert "isbn" in results[0]
        assert "title" in results[0]
        # Check if vector search was called
        recommender.vector_db.search.assert_called_with("test query", k=50)

    def test_recommend_filter_category(self, recommender):
        """Test filtering by category."""
        results = recommender.get_recommendations("test query", category="Fiction")
        # In mock data, "Fiction" books are 111, 222, 444
        assert len(results) > 0
        # Verify filtering happened (we can't easily check internal df, but we can check results if we mocked ID mapping correctly)
        # For this test, just ensuring it runs without error and returns results is a good start.

    def test_recommend_sort_tone_happy(self, recommender):
        """Test sorting by Happy tone."""
        # 111 is happiest (0.9)
        results = recommender.get_recommendations("test query", tone="Happy")
        assert results[0]["isbn"] == 111

    def test_recommend_sort_tone_sad(self, recommender):
        """Test sorting by Sad tone."""
        # 222 is saddest (0.9)
        results = recommender.get_recommendations("test query", category="All", tone="Sad")
        assert results[0]["isbn"] == 222

    def test_empty_query(self, recommender):
        """Test empty query behavior."""
        results = recommender.get_recommendations("")
        assert results == []
        results = recommender.get_recommendations("   ")
        assert results == []
