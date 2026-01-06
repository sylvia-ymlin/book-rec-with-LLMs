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
    # Mock search return value: list of objects with page_content attribute
    # The recommender expects "isbn description..." so we format it that way
    
    class MockDoc:
        def __init__(self, content):
            self.page_content = content
            
    mock_db.search.return_value = [
        MockDoc('111 "Description..."'),
        MockDoc('222 "Description..."'),
        MockDoc('333 "Description..."'),
        MockDoc('444 "Description..."'),
        MockDoc('555 "Description..."')
    ]
    return mock_db
