import pytest
from unittest.mock import MagicMock

from src.recommender import BookRecommender
from src.core.recommendation_orchestrator import RecommendationOrchestrator


def _mock_metadata_for_isbn(isbn: str, mock_books_df) -> dict:
    """Build metadata dict from mock_books_df for a given ISBN."""
    row = mock_books_df[mock_books_df["isbn13"].astype(str) == str(isbn)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "isbn13": str(r["isbn13"]),
        "title": r["title"],
        "authors": r["authors"],
        "description": r["description"],
        "simple_categories": r["simple_categories"],
        "joy": r["joy"],
        "sadness": r["sadness"],
        "fear": r["fear"],
        "anger": 0.1,
        "surprise": 0.1,
        "thumbnail": r["large_thumbnail"],
        "tags": "",
        "review_highlights": "",
        "average_rating": 4.0,
    }


class TestBookRecommender:
    @pytest.fixture
    def recommender(self, mock_books_df, mock_vector_db):
        """Initialize recommender with DI: inject mock_store and mock_vector_db. No patch needed."""
        mock_store = MagicMock()
        mock_store.get_book_metadata.side_effect = lambda isbn: _mock_metadata_for_isbn(isbn, mock_books_df)
        mock_store.get_all_categories.return_value = ["Fiction", "Non-Fiction", "Mystery"]

        orchestrator = RecommendationOrchestrator(
            metadata_store_inst=mock_store,
            vector_db=mock_vector_db,
        )
        return BookRecommender(orchestrator=orchestrator)

    def test_initialization(self, recommender):
        """Test if recommender initializes correctly (Zero-RAM mode: no in-memory books)."""
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
        results = recommender.get_recommendations_sync("test query")
        assert len(results) > 0
        assert "isbn" in results[0]
        assert "title" in results[0]
        # Check if vector search was called (hybrid_search is the default)
        recommender.vector_db.hybrid_search.assert_called()

    def test_recommend_filter_category(self, recommender):
        """Test filtering by category."""
        results = recommender.get_recommendations_sync("test query", category="Fiction")
        # In mock data, "Fiction" books are 111, 222, 444
        assert len(results) > 0
        # Verify filtering happened (we can't easily check internal df, but we can check results if we mocked ID mapping correctly)
        # For this test, just ensuring it runs without error and returns results is a good start.

    def test_recommend_sort_tone_happy(self, recommender):
        """Test sorting by Happy tone."""
        # 111 is happiest (0.9)
        results = recommender.get_recommendations_sync("test query", tone="Happy")
        assert str(results[0]["isbn"]) == "111"

    def test_recommend_sort_tone_sad(self, recommender):
        """Test Sad tone returns results (222 is saddest in mock data)."""
        results = recommender.get_recommendations_sync("test query", category="All", tone="Sad")
        assert len(results) > 0
        isbns = [str(r["isbn"]) for r in results]
        assert "222" in isbns  # Sad Book in mock

    def test_empty_query(self, recommender):
        """Test empty query behavior."""
        results = recommender.get_recommendations_sync("")
        assert results == []
        results = recommender.get_recommendations_sync("   ")
        assert results == []
