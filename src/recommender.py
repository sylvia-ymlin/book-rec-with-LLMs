"""
BookRecommender: Thin facade over RecommendationOrchestrator.
Preserves backward compatibility for main.py, agentic, tests, scripts.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from src.core.recommendation_orchestrator import RecommendationOrchestrator
from src.utils import setup_logger

logger = setup_logger(__name__)


class BookRecommender:
    """
    Facade: delegates all work to RecommendationOrchestrator.
    Kept for backward compatibility; new code may use RecommendationOrchestrator directly.
    Supports DI via orchestrator param for easier unit testing.
    """
    _orchestrator: RecommendationOrchestrator

    def __init__(self, orchestrator: RecommendationOrchestrator | None = None) -> None:
        self._orchestrator = orchestrator if orchestrator is not None else RecommendationOrchestrator()

    @property
    def vector_db(self):
        """Expose for main.py health check, benchmarks."""
        return self._orchestrator.vector_db

    @property
    def cache(self):
        return self._orchestrator.cache

    async def get_recommendations(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        user_id: str = "local",
        use_agentic: bool = False,
    ) -> List[Dict[str, Any]]:
        return await self._orchestrator.get_recommendations(
            query, category, tone, user_id, use_agentic
        )

    def get_recommendations_sync(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        user_id: str = "local",
        use_agentic: bool = False,
    ) -> List[Dict[str, Any]]:
        return self._orchestrator.get_recommendations_sync(
            query, category, tone, user_id, use_agentic
        )

    def get_similar_books(
        self,
        isbn: str,
        k: int = 10,
        category: str = "All",
    ) -> List[Dict[str, Any]]:
        return self._orchestrator.get_similar_books(isbn, k, category)

    def get_categories(self) -> List[str]:
        return self._orchestrator.get_categories()

    def get_tones(self) -> List[str]:
        return self._orchestrator.get_tones()

    def add_new_book(
        self,
        isbn: str,
        title: str,
        author: str,
        description: str,
        category: str = "General",
        thumbnail: Optional[str] = None,
        published_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return self._orchestrator.add_new_book(
            isbn, title, author, description, category, thumbnail, published_date
        )
