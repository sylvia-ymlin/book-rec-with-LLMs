"""
End-to-end integration tests.

Uses mocked recommender/rec_service to avoid loading real models.
Tests full request/response flow for key endpoints.
"""
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.app.main import app


def _mock_recommender():
    mock = MagicMock()
    mock.get_recommendations = AsyncMock(return_value=[
        {"isbn": "123", "title": "Book A", "authors": "Author", "description": "", "thumbnail": "", "caption": ""},
    ])
    mock.get_categories.return_value = ["All", "Fiction"]
    mock.get_similar_books.return_value = []
    mock.vector_db = MagicMock()
    return mock


def _mock_rec_service():
    mock = MagicMock()
    mock.get_recommendations.return_value = []
    mock.get_popular_books.return_value = []
    return mock


def test_e2e_recommend_flow():
    """Full flow: POST /recommend -> 200 -> recommendations in response."""
    mock_rec = _mock_recommender()
    mock_svc = _mock_rec_service()
    with patch("src.app.main.recommender", mock_rec), patch("src.app.main.rec_service", mock_svc):
        client = TestClient(app)
        response = client.post(
            "/recommend",
            json={"query": "thriller with plot twists", "category": "All", "user_id": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) >= 1
        book = data["recommendations"][0]
        assert "isbn" in book and "title" in book


def test_e2e_personal_recommend_flow():
    """Full flow: GET /api/recommend/personal -> 200."""
    mock_rec = _mock_recommender()
    mock_svc = _mock_rec_service()
    mock_svc.get_recommendations.return_value = [
        ("978123", 0.9, []),
        ("978456", 0.8, []),
    ]
    mock_rec.vector_db.get_book_details.return_value = {
        "title": "Test", "authors": "Auth", "description": "", "thumbnail": "",
    }
    with patch("src.app.main.recommender", mock_rec), patch("src.app.main.rec_service", mock_svc):
        client = TestClient(app)
        response = client.get("/api/recommend/personal?user_id=test&top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
