"""
Recommendation orchestrator: coordinates the recommendation flow only.
Delegates to VectorDB, Router, MetadataEnricher, FallbackProvider, Cache.
Single responsibility: flow coordination.
"""
from typing import Any, Dict, List, Optional

from src.config import TOP_K_INITIAL, TOP_K_FINAL
from src.vector_db import VectorDB
from src.cache import CacheManager
from src.core.metadata_store import metadata_store
from src.core.isbn_extractor import extract_isbn
from src.core.metadata_enricher import enrich_and_format
from src.core.fallback_provider import FallbackProvider
from src.core.book_ingestion import BookIngestion
from src.utils import setup_logger

logger = setup_logger(__name__)


class RecommendationOrchestrator:
    """
    Orchestrates RAG search and metadata enrichment.
    Zero business logic: only coordinates VectorDB, Router, Enricher, Fallback, Cache.
    Supports DI for metadata_store to simplify unit testing.
    """

    def __init__(
        self,
        metadata_store_inst=None,
        vector_db: Optional[VectorDB] = None,
        cache: Optional[CacheManager] = None,
        fallback_provider: Optional[FallbackProvider] = None,
        book_ingestion: Optional[BookIngestion] = None,
    ):
        self._meta = metadata_store_inst if metadata_store_inst is not None else metadata_store
        self.vector_db = vector_db or VectorDB()
        self.cache = cache or CacheManager()
        self._ingestion = book_ingestion or BookIngestion(
            vector_db=self.vector_db,
            metadata_store_inst=self._meta,
        )
        self._fallback = fallback_provider or FallbackProvider(
            book_ingestion=self._ingestion,
            metadata_store_inst=self._meta,
        )

        logger.info("RecommendationOrchestrator: Zero-RAM mode. Using SQLite for on-demand lookups.")

    async def get_recommendations(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        user_id: str = "local",
        use_agentic: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate book recommendations. Async for web search fallback.
        """
        if not query or not query.strip():
            return []

        cache_key = self.cache.generate_key("rec", q=query, c=category, t=tone, agentic=use_agentic)
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Returning cached results for key: {cache_key}")
            return cached

        logger.info(f"Processing request: query='{query}', category='{category}', use_agentic={use_agentic}")

        if use_agentic:
            results = await self._get_recommendations_agentic(query, category)
        else:
            results = await self._get_recommendations_classic(query, category)

        if results:
            self.cache.set(cache_key, results)
        return results

    def get_recommendations_sync(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        user_id: str = "local",
        use_agentic: bool = False,
    ) -> List[Dict[str, Any]]:
        """Sync wrapper for scripts/CLI."""
        import asyncio
        return asyncio.run(self.get_recommendations(query, category, tone, user_id, use_agentic))

    async def _get_recommendations_agentic(self, query: str, category: str) -> List[Dict[str, Any]]:
        """LangGraph workflow: Router -> Retrieve -> Evaluate -> (optional) Web Fallback."""
        from src.agentic.graph import get_agentic_graph

        graph = get_agentic_graph()
        config = {"configurable": {"recommender": self}}
        final_state = await graph.ainvoke(
            {"query": query, "category": category, "retry_count": 0},
            config=config,
        )
        books_list = final_state.get("isbn_list", [])
        return enrich_and_format(books_list, category, TOP_K_FINAL, "local", metadata_store_inst=self._meta)

    async def _get_recommendations_classic(self, query: str, category: str) -> List[Dict[str, Any]]:
        """Classic Router -> Hybrid/Small-to-Big -> optional Web Fallback."""
        from src.core.router import QueryRouter

        router = QueryRouter()
        decision = router.route(query)
        logger.info(f"Retrieval Strategy: {decision}")

        if decision["strategy"] == "small_to_big":
            recs = self.vector_db.small_to_big_search(query, k=TOP_K_INITIAL)
        else:
            recs = self.vector_db.hybrid_search(
                query,
                k=TOP_K_INITIAL,
                alpha=decision.get("alpha", 0.5),
                rerank=decision["rerank"],
                temporal=decision.get("temporal", False),
            )

        books_list = []
        for rec in recs:
            isbn_str = extract_isbn(rec)
            if isbn_str:
                books_list.append(isbn_str)

        results = enrich_and_format(books_list, category, TOP_K_FINAL, "local", metadata_store_inst=self._meta)

        if decision.get("freshness_fallback", False):
            threshold = decision.get("freshness_threshold", 3)
            if len(results) < threshold:
                web_results = await self._fallback.fetch_async(
                    query, TOP_K_FINAL - len(results), category
                )
                results.extend(web_results)
                logger.info(f"Web fallback added {len(web_results)} books")

        return results

    def get_similar_books(
        self,
        isbn: str,
        k: int = 10,
        category: str = "All",
    ) -> List[Dict[str, Any]]:
        """Content-based similar books by vector similarity."""
        isbn_str = str(isbn).strip()
        if not isbn_str:
            return []

        meta = self._meta.get_book_metadata(isbn_str)
        if not meta:
            logger.warning(f"get_similar_books: Book {isbn} not found in metadata")
            return []

        title = meta.get("title", "")
        description = meta.get("description", "") or ""
        if not title:
            logger.warning(f"get_similar_books: Book {isbn} has no title")
            return []

        query = f"{title} {description}"[:2000]
        recs = self.vector_db.search(query, k=k * 3)

        seen = {isbn_str}
        isbn_list = []
        for rec in recs:
            candidate = extract_isbn(rec)
            if candidate and candidate not in seen:
                seen.add(candidate)
                isbn_list.append(candidate)
            if len(isbn_list) >= k:
                break

        return enrich_and_format(isbn_list, category, k, "content_based", metadata_store_inst=self._meta)

    def get_categories(self) -> List[str]:
        """Get unique book categories."""
        return ["All"] + self._meta.get_all_categories()

    def get_tones(self) -> List[str]:
        """Get available emotional tones."""
        return ["All", "Happy", "Sad", "Fear", "Anger", "Surprise"]

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
        """Delegate to BookIngestion. Kept for agentic/facade compatibility."""
        return self._ingestion.add_book(
            isbn=isbn,
            title=title,
            author=author,
            description=description,
            category=category,
            thumbnail=thumbnail,
            published_date=published_date,
        )
