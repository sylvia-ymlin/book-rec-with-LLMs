from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch

# Mock the Recommender to avoid loading models during tests
with patch('src.main.BookRecommender') as MockRecommender:
    # Set up mock instance
    mock_instance = MockRecommender.return_value
    mock_instance.get_recommendations.return_value = [
        {
            "isbn": "123",
            "title": "Test Book",
            "authors": "Test Author",
            "description": "A test book description",
            "thumbnail": "test.jpg",
            "caption": "Test Book by Test Author"
        }
    ]
    mock_instance.get_categories.return_value = ["All", "Fiction"]
    mock_instance.get_tones.return_value = ["All", "Happy"]

    client = TestClient(app)

    def test_health():
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    # Note: Logic tests for /recommend are tricky because of the global recommender init.
    # In a real scenario, we'd use dependency injection.
    # For now, we tested that the app imports validly.
