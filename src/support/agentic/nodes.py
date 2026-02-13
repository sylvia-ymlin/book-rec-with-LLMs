"""
LangGraph nodes for the Agentic RAG workflow.
"""
from typing import Any, Dict

from src.support.agentic.state import RAGState
from src.infra.config import TOP_K_INITIAL
from src.rag.isbn_extractor import extract_isbn
from src.infra.utils import setup_logger

logger = setup_logger(__name__)


def router_node(state: RAGState) -> Dict[str, Any]:
    """Determine retrieval strategy using QueryRouter."""
    from src.rag.router import QueryRouter

    router = QueryRouter()
    decision = router.route(state["query"])
    logger.info(f"Agentic Router: {decision}")

    return {
        "strategy": decision["strategy"],
        "temporal": decision.get("temporal", False),
        "freshness_fallback": decision.get("freshness_fallback", False),
        "freshness_threshold": decision.get("freshness_threshold", 3),
        "decision_reason": f"routed to {decision['strategy']}",
    }


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """Execute retrieval based on strategy."""
    from src.rag.vector_db import VectorDB

    vector_db = VectorDB()
    strategy = state.get("strategy", "deep")
    query = state["query"]
    temporal = state.get("temporal", False)

    if strategy == "small_to_big":
        recs = vector_db.small_to_big_search(query, k=TOP_K_INITIAL)
    elif strategy == "exact":
        recs = vector_db.hybrid_search(
            query, k=TOP_K_INITIAL, alpha=1.0, rerank=False, temporal=False
        )
    else:
        recs = vector_db.hybrid_search(
            query,
            k=TOP_K_INITIAL,
            alpha=0.5,
            rerank=(strategy == "deep"),
            temporal=temporal,
        )

    isbn_list = []
    for doc in recs:
        isbn = extract_isbn(doc)
        if isbn:
            isbn_list.append(isbn)

    logger.info(f"Agentic Retrieve: {len(isbn_list)} results for strategy={strategy}")
    return {"isbn_list": isbn_list}


def evaluate_node(state: RAGState) -> Dict[str, Any]:
    """
    Evaluate if local results are sufficient (rule-based).
    Triggers web fallback when: few results + freshness query, or very few results.
    """
    n_results = len(state.get("isbn_list", []))
    freshness_fallback = state.get("freshness_fallback", False)
    threshold = state.get("freshness_threshold", 3)
    retry_count = state.get("retry_count", 0)

    # Hard limit: don't loop more than once
    if retry_count >= 1:
        return {"need_more": False}

    # Rule 1: No results and freshness query -> always need more
    if n_results == 0 and freshness_fallback:
        return {"need_more": True}

    # Rule 2: Results below threshold and freshness query -> need more
    if n_results < threshold and freshness_fallback:
        return {"need_more": True}

    # Rule 3: Very few results regardless -> need more
    if n_results < 2:
        return {"need_more": True}

    # Rule 4: Sufficient results
    return {"need_more": False}


async def web_fallback_node(state: RAGState, config=None) -> Dict[str, Any]:
    """
    Fetch from Google Books API when local results insufficient (async).
    Uses search_google_books_async to avoid blocking the event loop.
    """
    from src.rag.web_search import search_google_books_async
    from src.data.stores.metadata_store import metadata_store

    query = state["query"]
    category = state.get("category", "All")
    existing_isbns = set(state.get("isbn_list", []))
    max_to_fetch = 10 - len(existing_isbns)

    if max_to_fetch <= 0:
        return {"need_more": False}

    recommender = None
    if config:
        cfg = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {}) or {}
        recommender = cfg.get("recommender") if cfg else None

    web_books = await search_google_books_async(query, max_results=max_to_fetch * 2)
    new_isbns = list(existing_isbns)

    for book in web_books:
        isbn = book.get("isbn13", "")
        if not isbn or isbn in existing_isbns:
            continue
        if metadata_store.book_exists(isbn):
            continue
        if category and category != "All":
            book_cat = book.get("simple_categories", "")
            if category.lower() not in (book_cat or "").lower():
                continue

        if recommender:
            added = recommender.add_new_book(
                isbn=isbn,
                title=book.get("title", ""),
                author=book.get("authors", "Unknown"),
                description=book.get("description", ""),
                category=book.get("simple_categories", "General"),
                thumbnail=book.get("thumbnail"),
                published_date=book.get("publishedDate", ""),
            )
            if added:
                new_isbns.append(isbn)
        else:
            new_isbns.append(isbn)

        if len(new_isbns) - len(existing_isbns) >= max_to_fetch:
            break

    logger.info(f"Agentic Web Fallback: added {len(new_isbns) - len(existing_isbns)} books")
    return {"isbn_list": new_isbns, "need_more": False, "retry_count": 1}
