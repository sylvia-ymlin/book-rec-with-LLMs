import pytest
import pandas as pd
from unittest.mock import MagicMock

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
