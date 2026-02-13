"""
Handler for /api/recommend/personal endpoint.
Separates concerns: request parsing, intent probing, A/B config, result enrichment.
"""

from typing import Optional, List, Tuple, Any, Dict
from src.infra.utils import enrich_book_metadata, setup_logger

logger = setup_logger(__name__)


def parse_request_params(
    user_id: str,
    top_k: int,
    limit: Optional[int],
    recent_isbns: Optional[str],
    intent_query: Optional[str] = None,
) -> Tuple[str, int, Optional[List[str]]]:
    """
    Parse and normalize request parameters for personal recommendations.

    Demo logic: map 'local' to real user only when NOT cold-start (intent_query).
    Returns:
        (effective_user_id, k, real_time_seq)
    """
    k = limit if limit is not None else top_k
    effective_user_id = user_id
    if user_id in ["local", "demo"] and not intent_query:
        effective_user_id = "A1ZQ1LUQ9R6JHZ"

    real_time_seq = None
    if recent_isbns:
        real_time_seq = [x.strip() for x in recent_isbns.split(",") if x.strip()]

    return effective_user_id, k, real_time_seq


def resolve_seed_from_intent(
    intent_query: str,
    user_id: str,
    recommender: Any,
) -> Optional[List[str]]:
    """
    P2: Zero-shot intent probing — when no recent_isbns, use query to seed.
    Probes LLM for categories/keywords, does semantic search, returns seed ISBNs.

    Returns:
        List of seed ISBNs, or None if probing failed or produced no results.
    """
    if not intent_query or not intent_query.strip():
        return None

    from src.rag.intent_prober import probe_intent

    intent = probe_intent(intent_query.strip())
    semantic_query = " ".join(
        intent.get("keywords", []) + intent.get("categories", []) + [intent.get("summary", "")]
    ).strip()

    if not semantic_query or not recommender:
        return None

    try:
        rag_results = recommender.get_recommendations_sync(
            semantic_query, category="All", tone="All", user_id=user_id
        )
        seed_isbns = [r.get("isbn") for r in (rag_results or [])[:5] if r.get("isbn")]
        return seed_isbns if seed_isbns else None
    except Exception as e:
        logger.warning(f"Intent-to-seed failed: {e}")
        return None


def get_ab_diversity_config(
    user_id: str,
    experiment_id: Optional[str],
    ab_variant: Optional[str],
) -> bool:
    """
    Resolve A/B experiment config for diversity rerank.
    Returns enable_diversity (True = treatment, False = control).
    """
    enable_diversity = True
    if experiment_id:
        from src.core.ab_experiments import get_experiment_config, log_experiment
        from src.infra.config import AB_EXPERIMENTS_ENABLED

        if AB_EXPERIMENTS_ENABLED:
            cfg = get_experiment_config(user_id, experiment_id, ab_variant)
            enable_diversity = cfg.get("enable_diversity_rerank", True)
            variant = "treatment" if enable_diversity else "control"
            log_experiment(experiment_id, user_id, variant)
    return enable_diversity


def enrich_personal_results(
    recs: List[Tuple[str, float, list]],
    get_book_details: Any,
) -> List[Dict[str, Any]]:
    """
    Enrich raw (isbn, score, explanation) tuples into display-ready dicts.
    Fetches metadata, covers, formats tags/highlights.
    """
    results = []
    for isbn, score, explanation in recs:
        meta = get_book_details(isbn) or {}
        meta = enrich_book_metadata(meta, str(isbn))

        title = meta.get("title") or f"ISBN: {isbn}"
        desc = meta.get("description", "No description available.")
        thumb = meta.get("thumbnail", "/content/cover-not-found.jpg") or "/content/cover-not-found.jpg"
        authors = meta.get("authors", "Unknown")

        rating = 0.0
        if meta:
            rating = float(meta.get("average_rating", meta.get("rating", 0.0)))

        tags = []
        if meta and "tags" in meta:
            tags_raw = meta["tags"]
            if isinstance(tags_raw, str):
                tags = [t.strip() for t in tags_raw.split(";") if t.strip()]
            elif isinstance(tags_raw, list):
                tags = tags_raw

        highlights = []
        if meta and "review_highlights" in meta:
            h_raw = meta["review_highlights"]
            if isinstance(h_raw, str):
                highlights = [h.strip() for h in h_raw.split(";") if h.strip()][:3]

        results.append({
            "isbn": isbn,
            "score": float(score),
            "title": title,
            "authors": authors,
            "description": desc,
            "thumbnail": thumb,
            "average_rating": rating,
            "tags": tags,
            "review_highlights": highlights,
            "caption": f"{title} by {authors}",
            "explanations": explanation,
        })
    return results
