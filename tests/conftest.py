import pytest
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture(scope="session", autouse=True)
def mock_main_startup():
    """Prevent main startup from loading real models (ensure_models_exist, RecommendationOrchestrator).
    Applied before any test; first request triggers startup which uses mocks."""
    mock_rec = MagicMock()
    mock_rec.get_recommendations = AsyncMock(return_value=[])
    mock_rec.get_categories.return_value = ["All", "Fiction"]
    mock_rec.get_tones.return_value = ["All", "Happy"]
    mock_rec.get_similar_books.return_value = []
    mock_rec.vector_db = MagicMock()
    mock_svc = MagicMock()
    mock_svc.get_recommendations.return_value = []
    mock_svc.get_popular_books.return_value = []
    mock_svc.load_resources = MagicMock()
    with patch("src.core.model_loader.ensure_models_exist"), \
         patch("src.app.state.recommender", mock_rec), \
         patch("src.app.state.rec_service", mock_svc):
        yield


@pytest.fixture
def mock_books_df():
    """Create a mock DataFrame for books."""
    data = {
        "isbn13": [111, 222, 333, 444, 555],
        "title": ["Happy Book", "Sad Book", "Scary Book", "Fiction Book", "Non-Fiction Book"],
        "authors": "Author A;Author B",
        "description": "This is a test description " * 5,
        "simple_categories": ["Fiction", "Fiction", "Mystery", "Fiction", "Non-Fiction"],
        "joy": [0.9, 0.1, 0.1, 0.5, 0.5],
        "sadness": [0.1, 0.9, 0.1, 0.2, 0.2],
        "fear": [0.1, 0.1, 0.9, 0.1, 0.1],
        "large_thumbnail": ["http://example.com/img1.jpg"] * 5
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_vector_db():
    """Mock the VectorDB singleton."""
    mock_db = MagicMock()
    
    class MockDoc:
        def __init__(self, content, metadata=None):
            self.page_content = content
            self.metadata = metadata or {}
            
    docs = [
        MockDoc('111 "Happy Book"', {"isbn": "111", "title": "Happy Book"}),
        MockDoc('222 "Sad Book"', {"isbn": "222", "title": "Sad Book"}),
        MockDoc('333 "Scary Book"', {"isbn": "333", "title": "Scary Book"}),
        MockDoc('444 "Fiction Book"', {"isbn": "444", "title": "Fiction Book"}),
        MockDoc('555 "Non-Fiction Book"', {"isbn": "555", "title": "Non-Fiction Book"})
    ]
    
    # Mock all potential search methods
    mock_db.search.return_value = docs
    mock_db.hybrid_search.return_value = docs
    mock_db.small_to_big_search.return_value = docs
    
    return mock_db
