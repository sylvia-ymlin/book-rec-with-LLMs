"""
Integration tests for FastAPI endpoints.

Mocks src.app.main.recommender and rec_service so endpoints use fakes without loading models.
"""
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.app.main import app


def _make_mock_recommender():
    """Create mock recommender with all endpoint needs."""
    mock = MagicMock()
    mock.get_recommendations = AsyncMock(return_value=[
        {
            "isbn": "123",
            "title": "Test Book",
            "authors": "Test Author",
            "description": "A test book description",
            "thumbnail": "test.jpg",
            "caption": "Test Book by Test Author",
        }
    ])
    mock.get_categories.return_value = ["All", "Fiction"]
    mock.get_tones.return_value = ["All", "Happy"]
    mock.get_similar_books.return_value = []
    mock.vector_db = MagicMock()
    return mock


def _make_mock_rec_service():
    """Create mock rec_service for personalized endpoints."""
    mock = MagicMock()
    mock.get_recommendations.return_value = []
    mock.get_popular_books.return_value = []
    return mock


def test_health():
    """Health check returns 200 and healthy status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_recommend_mocked():
    """POST /recommend returns recommendations when recommender is mocked."""
    mock_rec = _make_mock_recommender()
    mock_svc = _make_mock_rec_service()
    with patch("src.app.main.recommender", mock_rec), patch("src.app.main.rec_service", mock_svc):
        client = TestClient(app)
        response = client.post(
            "/recommend",
            json={"query": "a romantic comedy", "category": "All"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        assert data["recommendations"][0]["isbn"] == "123"


def test_categories_mocked():
    """GET /categories returns category list."""
    mock_rec = _make_mock_recommender()
    mock_svc = _make_mock_rec_service()
    with patch("src.app.main.recommender", mock_rec), patch("src.app.main.rec_service", mock_svc):
        client = TestClient(app)
        response = client.get("/categories")
        assert response.status_code == 200
        assert "categories" in response.json()
        assert "All" in response.json()["categories"]


def test_recommend_empty_query():
    """POST /recommend with empty query returns empty or error."""
    mock_rec = _make_mock_recommender()
    mock_rec.get_recommendations = AsyncMock(return_value=[])
    mock_svc = _make_mock_rec_service()
    with patch("src.app.main.recommender", mock_rec), patch("src.app.main.rec_service", mock_svc):
        client = TestClient(app)
        response = client.post(
            "/recommend",
            json={"query": "", "category": "All"},
        )
        assert response.status_code == 200
        assert response.json()["recommendations"] == []


def test_similar_books_mocked():
    """GET /api/recommend/similar/{isbn} delegates to recommender."""
    mock_rec = _make_mock_recommender()
    mock_rec.get_similar_books.return_value = [
        {"isbn": "456", "title": "Similar Book", "authors": "Author B", "description": "", "thumbnail": ""}
    ]
    mock_svc = _make_mock_rec_service()
    with patch("src.app.main.recommender", mock_rec), patch("src.app.main.rec_service", mock_svc):
        client = TestClient(app)
        response = client.get("/api/recommend/similar/123?k=5")
        assert response.status_code == 200
        assert response.json()["recommendations"][0]["isbn"] == "456"


def test_metrics_endpoint():
    """GET /metrics returns Prometheus format."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or "HELP" in response.text
