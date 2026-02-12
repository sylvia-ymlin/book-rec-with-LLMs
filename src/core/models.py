"""
Domain models for type safety.

Replaces Dict[str, Any] in core recommendation flow with typed structures.
Enables IDE autocomplete and static type checking.
"""
from typing import TypedDict, Optional, List, Any, Tuple


# -----------------------------------------------------------------------------
# Book / Metadata types
# -----------------------------------------------------------------------------


class BookMetadata(TypedDict, total=False):
    """
    Book metadata from MetadataStore / OnlineBooksStore.
    total=False: all fields optional to support partial data from various sources.
    """
    isbn13: str
    isbn10: Optional[str]
    title: str
    authors: str
    description: Optional[str]
    simple_categories: Optional[str]
    thumbnail: Optional[str]
    image: Optional[str]
    average_rating: Optional[float]
    publishedDate: Optional[str]
    # Emotion scores (from books_processed)
    joy: float
    sadness: float
    fear: float
    anger: float
    surprise: float
    tags: Optional[str]
    review_highlights: Optional[str]
    # For compatibility with extra DB columns
    large_thumbnail: Optional[str]


class BookResponseDict(TypedDict, total=False):
    """Standard API response structure for a single book recommendation."""
    isbn: str
    title: str
    authors: str
    description: str
    thumbnail: Optional[str]
    caption: str
    tags: List[str]
    emotions: dict
    review_highlights: List[str]
    persona_summary: str
    average_rating: float
    source: str


# -----------------------------------------------------------------------------
# Router / Orchestration types
# -----------------------------------------------------------------------------


class RouterDecision(TypedDict, total=False):
    """
    Query router output: retrieval strategy and parameters.
    Used by RecommendationOrchestrator to decide hybrid vs small-to-big, rerank, etc.
    """
    strategy: str  # 'exact' | 'fast' | 'deep' | 'small_to_big'
    alpha: float  # Hybrid search weight (0=dense-only, 1=sparse-only)
    rerank: bool  # Apply cross-encoder reranking
    k_final: int  # Number of results to return
    temporal: bool  # Apply temporal decay boost
    freshness_fallback: bool  # Enable web search when local results insufficient
    freshness_threshold: int  # Min local results before triggering web fallback
    target_year: Optional[int]  # Specific year user requested (e.g. 2024)


# -----------------------------------------------------------------------------
# Recall / Ranking types
# -----------------------------------------------------------------------------


# Type alias: (isbn, score, explanations) - DiversityReranker input/output
ScoredCandidateTuple = Tuple[str, float, list]
