"""
Recommendation orchestrator: coordinates the recommendation flow only.
Delegates to VectorDB, Router, MetadataEnricher, FallbackProvider, Cache.
Single responsibility: flow coordination.
"""
from typing import Any, List, Optional, Tuple

from src.infra.config import (
    TOP_K_INITIAL,
    TOP_K_FINAL,
    ENABLE_RAG_DIVERSITY,
    DATA_DIR,
    MMR_LAMBDA_DEFAULT,
    POPULARITY_GAMMA_DEFAULT,
    MAX_PER_CATEGORY_DEFAULT,
    HYBRID_DEFAULT_ALPHA,
    FRESHNESS_THRESHOLD,
)
from src.core.models import BookResponseDict, RouterDecision
from src.rag.vector_db import VectorDB
from src.infra.cache import CacheManager
from src.data.stores.metadata_store import metadata_store
from src.rag.isbn_extractor import extract_isbn
from src.core.metadata_enricher import enrich_and_format
from src.rag.fallback_provider import FallbackProvider
from src.core.book_ingestion import BookIngestion
from src.infra.utils import setup_logger

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
        fast: bool = False,
        async_rerank: bool = False,
        enable_diversity_rerank: bool = True,
    ) -> List[BookResponseDict]:
        """
        Generate book recommendations. Async for web search fallback.
        fast: Skip rerank for low latency (~150ms).
        async_rerank: Return RRF immediately, rerank in background; next request gets cached reranked.
        enable_diversity_rerank: Apply MMR + category diversity to RAG results (when ENABLE_RAG_DIVERSITY).
        """
        if not query or not query.strip():
            return []

        cache_key = self.cache.generate_key(
            "rec", q=query, c=category, t=tone, agentic=use_agentic, div=enable_diversity_rerank
        )
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Returning cached results for key: {cache_key}")
            return cached

        logger.info(f"Processing request: query='{query}', category='{category}', use_agentic={use_agentic}, fast={fast}, async_rerank={async_rerank}")

        skip_rerank = fast or async_rerank

        if use_agentic:
            results = await self._get_recommendations_agentic(query, category)
        else:
            results = await self._get_recommendations_classic(
                query, category, skip_rerank=skip_rerank, enable_diversity_rerank=enable_diversity_rerank
            )

        if results:
            self.cache.set(cache_key, results)

        if async_rerank and not use_agentic and skip_rerank:
            import asyncio
            asyncio.create_task(
                self._background_rerank_and_cache(query, category, cache_key, enable_diversity_rerank)
            )

        return results

    async def _background_rerank_and_cache(
        self, query: str, category: str, cache_key: str, enable_diversity_rerank: bool = True
    ) -> None:
        """Run full pipeline with rerank and cache for async_rerank flow."""
        try:
            results = await self._get_recommendations_classic(
                query, category, skip_rerank=False, enable_diversity_rerank=enable_diversity_rerank
            )
            if results:
                self.cache.set(cache_key, results)
                logger.info(f"Background rerank completed for query '{query[:30]}...'")
        except Exception as e:
            logger.warning(f"Background rerank failed [{type(e).__name__}]: {e}")

    def get_recommendations_sync(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        user_id: str = "local",
        use_agentic: bool = False,
        fast: bool = False,
        async_rerank: bool = False,
        enable_diversity_rerank: bool = True,
    ) -> List[BookResponseDict]:
        """Sync wrapper for scripts/CLI."""
        import asyncio
        return asyncio.run(
            self.get_recommendations(
                query, category, tone, user_id, use_agentic, fast, async_rerank, enable_diversity_rerank
            )
        )

    async def _get_recommendations_agentic(self, query: str, category: str) -> List[BookResponseDict]:
        """LangGraph workflow: Router -> Retrieve -> Evaluate -> (optional) Web Fallback."""
        from src.support.agentic.graph import get_agentic_graph

        graph = get_agentic_graph()
        config = {"configurable": {"recommender": self}}
        final_state = await graph.ainvoke(
            {"query": query, "category": category, "retry_count": 0},
            config=config,
        )
        books_list = final_state.get("isbn_list", [])
        return enrich_and_format(books_list, category, TOP_K_FINAL, "local", metadata_store_inst=self._meta)

    def _recs_to_scored_tuples(self, recs: List[Any]) -> List[Tuple[str, float, list]]:
        """Convert vector search results to (isbn, score, []) for DiversityReranker."""
        tuples: List[Tuple[str, float, list]] = []
        for rank, rec in enumerate(recs):
            isbn_str = extract_isbn(rec)
            if not isbn_str:
                continue
            score = 0.0
            if hasattr(rec, "metadata") and rec.metadata:
                score = float(rec.metadata.get("relevance_score", 0))
            if score <= 0:
                score = 1.0 / (rank + 1)  # Rank-based when no reranker score
            tuples.append((isbn_str, score, []))
        return tuples

    async def _get_recommendations_classic(
        self,
        query: str,
        category: str,
        skip_rerank: bool = False,
        enable_diversity_rerank: bool = True,
    ) -> List[BookResponseDict]:
        """Classic Router -> Hybrid/Small-to-Big -> optional Diversity Rerank -> Web Fallback."""
        from src.rag.router import QueryRouter as _QueryRouter

        router = _QueryRouter()
        decision = router.route(query)
        logger.info(f"Retrieval Strategy: {decision}")

        do_rerank = decision["rerank"] and not skip_rerank

        if decision["strategy"] == "small_to_big":
            recs = self.vector_db.small_to_big_search(query, k=TOP_K_INITIAL)
        else:
            recs = self.vector_db.hybrid_search(
                query,
                k=TOP_K_INITIAL,
                alpha=decision.get("alpha", HYBRID_DEFAULT_ALPHA),
                rerank=do_rerank,
                temporal=decision.get("temporal", False),
            )

        scored_tuples = self._recs_to_scored_tuples(recs)
        books_list = [t[0] for t in scored_tuples]

        # RAG diversity: MMR + category constraint (when enabled)
        if ENABLE_RAG_DIVERSITY and enable_diversity_rerank and scored_tuples:
            from src.core.diversity_reranker import DiversityReranker
            data_dir = str(DATA_DIR / "rec")
            div_reranker = DiversityReranker(
                metadata_store=self._meta,
                data_dir=data_dir,
                mmr_lambda=MMR_LAMBDA_DEFAULT,
                popularity_gamma=POPULARITY_GAMMA_DEFAULT,
                max_per_category=MAX_PER_CATEGORY_DEFAULT,
            )
            reranked = div_reranker.rerank(scored_tuples, top_k=TOP_K_FINAL * 2)
            books_list = [t[0] for t in reranked]

        results = enrich_and_format(books_list, category, TOP_K_FINAL, "local", metadata_store_inst=self._meta)

        if decision.get("freshness_fallback", False):
            threshold = decision.get("freshness_threshold", FRESHNESS_THRESHOLD)
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
    ) -> List[BookResponseDict]:
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
    ) -> Optional[BookResponseDict]:
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
