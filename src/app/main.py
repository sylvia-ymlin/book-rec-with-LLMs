from fastapi import FastAPI, HTTPException, Request, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import time

import prometheus_client
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.core.recommendation_orchestrator import RecommendationOrchestrator
from src.utils import setup_logger
from src.data.stores.profile_store import (
    add_favorite,
    list_favorites,
    remove_favorite,
    update_book_rating,
    update_reading_status,
    update_book_comment,
    get_favorites_with_metadata,
    get_reading_stats,
)
from src.app.api.chat import router as chat_router
from src.services.chat_service import chat_service
from src.services.recommend_service import RecommendationService
from src.services.personal_recommend_handler import (
    parse_request_params,
    resolve_seed_from_intent,
    get_ab_diversity_config,
    enrich_personal_results,
)


logger = setup_logger(__name__)

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total count of HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)


app = FastAPI(
    title="Book Recommender API",
    description="""Intelligent Book Recommendation System with RAG + Personalized RecSys.

## Overview
- **RAG Path**: Semantic search (BM25 + Dense) → Router → Rerank for vague queries
- **RecSys Path**: 7-channel recall → LGBMRanker for personalized recommendations
- **Chat**: Stream LLM responses with book context (RAG)

## Quick Links
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
""",
    version="2.6.0",
)

# Include Routers
app.include_router(chat_router)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    method = request.method
    path = request.url.path

    # Skip noise endpoints
    if path in ["/metrics", "/health"]:
        return await call_next(request)

    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        status = "500"
        raise
    finally:
        process_time = time.perf_counter() - start_time
        REQUEST_COUNT.labels(method=method, endpoint=path, status_code=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(process_time)

    return response


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Expose Prometheus metrics (request count, latency histograms). Scrape at `/metrics`.",
    responses={200: {"description": "Prometheus text format"}},
)
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Initialize Recommender and Services (Singleton)
# We do this on startup so the first request is fast
recommender = None
rec_service = None


@app.on_event("startup")
async def startup_event():
    global recommender, rec_service

    # Download models from HF Hub if not present (for HF Spaces deployment)
    from src.core.model_loader import ensure_models_exist

    logger.info("Checking/downloading models from HF Hub...")
    ensure_models_exist()

    logger.info("Initializing Recommender Engine...")
    from src.data.stores.metadata_store import metadata_store as _metadata_store

    recommender = RecommendationOrchestrator(metadata_store_inst=_metadata_store)

    logger.info("Initializing Personalized Rec Service...")
    rec_service = RecommendationService()
    # Pre-warm resources for better UX
    try:
        rec_service.load_resources()
    except Exception as e:
        logger.error(f"Failed to pre-load resources: {e}")

    logger.info("Engines Initialized.")


class RecommendationRequest(BaseModel):
    """Request body for semantic + RAG-based recommendations."""

    query: str = Field(..., description="Natural language query (e.g. 'a thriller with plot twists')")
    category: str = Field(default="All", description="Filter by category (e.g. Fiction, Romance)")
    user_id: Optional[str] = Field(default="local", description="User identifier for personalization")
    use_agentic: Optional[bool] = Field(
        default=False,
        description="Enable LangGraph workflow: Router → Retrieve → Evaluate → Web Fallback",
    )
    fast: Optional[bool] = Field(
        default=False,
        description="Skip rerank for ~150ms latency (RRF only)",
    )
    async_rerank: Optional[bool] = Field(
        default=False,
        description="Return RRF first, rerank in background; next request gets cached",
    )
    experiment_id: Optional[str] = Field(default=None, description="A/B experiment ID for variant assignment")
    ab_variant: Optional[str] = Field(default=None, description="Force variant: 'control' | 'treatment'")

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "a romantic comedy set in New York", "category": "Fiction"}]
        }
    }


class FeatureContribution(BaseModel):
    """SHAP feature contribution for explainability."""

    feature: str = Field(..., description="Feature name (e.g. 'title_similarity')")
    contribution: float = Field(..., description="Contribution score")
    direction: str = Field(..., description="'positive' or 'negative'")


class BookResponse(BaseModel):
    """Single book in recommendation response."""

    isbn: str = Field(..., description="ISBN-10 or ISBN-13")
    title: str
    authors: str
    description: str
    thumbnail: str = Field(..., description="Cover image URL or path")
    caption: str = Field(default="", description="One-line summary")
    tags: List[str] = Field(default_factory=list)
    average_rating: float = Field(default=0.0, ge=0, le=5)
    explanations: List[FeatureContribution] = Field(
        default_factory=list, description="SHAP explanations"
    )


class RecommendationResponse(BaseModel):
    """Response with list of recommended books."""

    recommendations: List[BookResponse] = Field(
        ..., description="Ordered list of recommended books"
    )


class FavoriteRequest(BaseModel):
    """Request for add/remove favorite."""

    user_id: Optional[str] = Field(default="local")
    isbn: str = Field(..., description="ISBN of the book")


class HighlightsRequest(BaseModel):
    """Request for book highlights."""

    isbn: str
    user_id: Optional[str] = Field(default="local")


class BookUpdateRequest(BaseModel):
    """Update rating, reading status, or comment for a favorited book."""

    user_id: Optional[str] = Field(default="local")
    isbn: str
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    status: Optional[str] = Field(
        default=None, description="'want_to_read' | 'reading' | 'finished'"
    )
    comment: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [{"isbn": "0140283331", "rating": 4.5, "status": "finished"}]
        }
    }


class BookAddRequest(BaseModel):
    """Add a new book to the database and vector index."""

    isbn: str = Field(..., description="Unique ISBN (10 or 13 digits)")
    title: str
    author: str
    description: str
    category: Optional[str] = Field(default="General")
    thumbnail: Optional[str] = Field(default=None, description="Cover image URL")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "isbn": "9780140283337",
                    "title": "Catcher in the Rye",
                    "author": "J.D. Salinger",
                    "description": "A novel about teenage alienation.",
                }
            ]
        }
    }


@app.post(
    "/books/add",
    summary="Add book",
    description="Add a new book to SQLite metadata store and Chroma vector index. ISBN must be unique.",
    responses={
        200: {"description": "Book added successfully"},
        400: {"description": "ISBN already exists or invalid"},
        500: {"description": "Internal error"},
        503: {"description": "Service not ready"},
    },
)
async def add_book_endpoint(req: BookAddRequest):
    """Dynamically add a new book to the database and vector index."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        new_book_row = recommender.add_new_book(
            req.isbn,
            req.title,
            req.author,
            req.description,
            req.category,
            req.thumbnail,
        )
        if new_book_row is not None:
            # Also update ChatService context
            chat_service.add_book_to_context(new_book_row)
            return {"status": "success", "message": f"Book {req.isbn} added."}
        msg = "Failed to add book. Ensure ISBN is unique."
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.error(f"Error adding book: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/health",
    summary="Health check",
    description="Verify service is running. Returns `{status: 'healthy'}`.",
    responses={200: {"description": "Service healthy"}},
)
async def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "healthy"}


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Semantic recommendations",
    description="""Generate book recommendations via RAG pipeline:
- **Hybrid search**: BM25 + Dense vector, RRF fusion
- **Rerank**: Cross-Encoder or ONNX (~2x faster)
- **use_agentic=true**: LangGraph Router → Retrieve → Evaluate → Web Fallback (for vague queries)
- **fast=true**: Skip rerank (~150ms latency)
- **async_rerank=true**: Return RRF first, rerank in background; next request gets cached
- **Response**: `{recommendations: [{isbn, title, authors, description, thumbnail, ...}]}`
""",
    responses={
        200: {"description": "List of recommended books"},
        500: {"description": "Processing error"},
        503: {"description": "Service not ready"},
    },
)
async def get_recommendations(request: RecommendationRequest):
    """Generate book recommendations based on semantic search and emotion/category filtering."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        user_id = request.user_id if hasattr(request, "user_id") else "local"
        enable_diversity = True  # default
        if request.experiment_id:
            from src.core.ab_experiments import get_experiment_config, log_experiment
            from src.config import AB_EXPERIMENTS_ENABLED

            if AB_EXPERIMENTS_ENABLED:
                cfg = get_experiment_config(
                    user_id, request.experiment_id, request.ab_variant
                )
                enable_diversity = cfg.get("enable_diversity_rerank", True)
                variant = "treatment" if enable_diversity else "control"
                log_experiment(request.experiment_id, user_id, variant)
        results = await recommender.get_recommendations(
            query=request.query,
            category=request.category,
            user_id=user_id,
            use_agentic=request.use_agentic or False,
            fast=request.fast or False,
            async_rerank=request.async_rerank or False,
            enable_diversity_rerank=enable_diversity,
        )
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/recommend/similar/{isbn}",
    response_model=RecommendationResponse,
    summary="Similar books by ISBN",
    description="""Content-based similar books by vector similarity.
- **Use case**: User clicks a book → show similar recommendations immediately
- **No user history required**: Works for new users and new books in ChromaDB
- **Params**: `k` (default 10), `category` (default All)
- **Response**: `{recommendations: [{isbn, title, authors, ...}]}`
""",
    responses={
        200: {"description": "List of similar books"},
        500: {"description": "Error"},
        503: {"description": "Service not ready"},
    },
)
def get_similar_books(
    isbn: str = Path(..., description="ISBN of the seed book"),
    k: int = Query(10, ge=1, le=50, description="Number of similar books to return"),
    category: str = Query("All", description="Filter by category"),
):
    """Content-based similar books by vector similarity."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        results = recommender.get_similar_books(isbn=isbn, k=k, category=category)
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"get_similar_books error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/categories",
    summary="List categories",
    description="Return all book categories available for filtering. Response: `{categories: ['Fiction', 'Romance', ...]}`.",
    responses={200: {"description": "List of category names"}, 503: {"description": "Service not ready"}},
)
async def get_categories():
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"categories": recommender.get_categories()}


@app.post(
    "/favorites/add",
    summary="Add favorite",
    description="Add a book to user's favorites. Response: `{status: 'ok', favorites_count: N}`.",
    responses={
        200: {"description": "Added"},
        500: {"description": "Error"},
        503: {"description": "Service not ready"},
    },
)
async def favorites_add(req: FavoriteRequest):
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        count = add_favorite(req.user_id or "local", req.isbn)
        return {"status": "ok", "favorites_count": count}
    except Exception as e:
        logger.error(f"favorites_add error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/favorites/remove",
    summary="Remove favorite",
    description="Remove a book from user's favorites. Response: `{status: 'ok', favorites_count: N}`.",
    responses={
        200: {"description": "Removed"},
        500: {"description": "Error"},
        503: {"description": "Service not ready"},
    },
)
async def favorites_remove(req: FavoriteRequest):
    """Remove a book from user's favorites."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        count = remove_favorite(req.user_id or "local", req.isbn)
        return {"status": "ok", "favorites_count": count}
    except Exception as e:
        logger.error(f"favorites_remove error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put(
    "/favorites/update",
    summary="Update favorite",
    description="""Update rating, reading status, or comment for a favorited book.
- **rating**: 1–5 (optional)
- **status**: `want_to_read` | `reading` | `finished` (optional)
- **comment**: Free text (optional)
- Response: `{status: 'ok'}`. All fields optional; only provided fields are updated.
""",
    responses={
        200: {"description": "Updated"},
        500: {"description": "Error"},
        503: {"description": "Service not ready"},
    },
)
async def favorites_update(req: BookUpdateRequest):
    """Update rating, status, or comment for a favorited book."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        uid = req.user_id or "local"
        if req.rating is not None:
            update_book_rating(uid, req.isbn, req.rating)
        if req.status is not None:
            update_reading_status(uid, req.isbn, req.status)
        if req.comment is not None:
            update_book_comment(uid, req.isbn, req.comment)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"favorites_update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/favorites/list/{user_id}",
    summary="List favorites",
    description="""Return user's favorite books with full details.
- **Response**: `{favorites: [{isbn, title, author, img, category, rating, status, added_at, comment}]}`
- **status**: `want_to_read` | `reading` | `finished`
""",
    responses={
        200: {"description": "List of favorites"},
        500: {"description": "Error"},
        503: {"description": "Service not ready"},
    },
)
async def favorites_list(user_id: str):
    """Return user's favorite books with full details."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        # Get favorites with metadata (rating, status)
        favorites_meta = get_favorites_with_metadata(user_id)
        # ENGINEERING IMPROVEMENT: Zero-RAM Lookup
        from src.data.stores.metadata_store import metadata_store
        from src.utils import enrich_book_metadata

        results = []
        for isbn, meta in favorites_meta.items():
            book_meta = metadata_store.get_book_metadata(str(isbn))

            # 1. Enrich (fetch covers if needed)
            book_meta = enrich_book_metadata(book_meta, str(isbn))

            # 2. Extract Display Fields
            title = book_meta.get("title") or f"Unknown Book ({isbn})"
            thumbnail = book_meta.get("thumbnail") or "/content/cover-not-found.jpg"
            author = book_meta.get("authors", "Unknown")

            results.append(
                {
                    "isbn": isbn,
                    "title": title,
                    "author": author,
                    "img": thumbnail,
                    "category": book_meta.get("simple_categories", ""),
                    "rating": meta.get("rating"),
                    "status": meta.get("status", "want_to_read"),
                    "added_at": meta.get("added_at"),
                    "comment": meta.get("comment", ""),
                }
            )
        return {"favorites": results}
    except Exception as e:
        logger.error(f"favorites_list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/benchmark",
    summary="Performance benchmark",
    description="""Run performance benchmark (5 queries). Returns latency stats for vector search and full recommendation.
- **Response**: `{vector_search: {mean_ms, median_ms, ...}, full_recommendation: {...}, dataset_size: N}`
""",
    responses={
        200: {"description": "Benchmark results"},
        503: {"description": "Service not ready"},
    },
)
async def run_benchmark():
    """Run performance benchmark and return latency metrics."""
    import statistics

    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")

    test_queries = [
        "a romantic comedy set in New York",
        "a philosophical novel about the meaning of life",
        "a fast-paced thriller with plot twists",
        "a coming-of-age story about friendship",
        "a science fiction story about space exploration",
    ]

    # Benchmark vector search
    vector_latencies = []
    for query in test_queries:
        start = time.perf_counter()
        recommender.vector_db.search(query, k=50)
        vector_latencies.append((time.perf_counter() - start) * 1000)

    # Benchmark full recommendation (async)
    full_latencies = []
    for query in test_queries:
        start = time.perf_counter()
        await recommender.get_recommendations(query, "All", "All")
        full_latencies.append((time.perf_counter() - start) * 1000)

    # Estimate size
    size = 20000
    if recommender.vector_db.db:
        size = recommender.vector_db.db._collection.count()

    return {
        "vector_search": {
            "runs": len(vector_latencies),
            "mean_ms": round(statistics.mean(vector_latencies), 2),
            "median_ms": round(statistics.median(vector_latencies), 2),
            "min_ms": round(min(vector_latencies), 2),
            "max_ms": round(max(vector_latencies), 2),
        },
        "full_recommendation": {
            "runs": len(full_latencies),
            "mean_ms": round(statistics.mean(full_latencies), 2),
            "median_ms": round(statistics.median(full_latencies), 2),
            "min_ms": round(min(full_latencies), 2),
            "max_ms": round(max(full_latencies), 2),
        },
        "dataset_size": size,
    }


@app.get(
    "/api/recommend/personal",
    response_model=RecommendationResponse,
    summary="Personalized recommendations",
    description="""Get personalized recommendations from reading history. Uses 6-channel recall (ItemCF/UserCF/Swing/SASRec/YoutubeDNN/Popularity) + LGBMRanker.

| Param | Type | Description |
|-------|------|--------------|
| user_id | str | User identifier (default: local) |
| top_k | int | Number of recommendations (default: 10) |
| limit | int | Alias for top_k (optional) |
| recent_isbns | str | Comma-separated ISBNs from current session (e.g. just-viewed). Injected into SASRec for cold-start. |
| intent_query | str | Zero-shot intent probing when no history. LLM infers categories/keywords → semantic search → seeds SASRec. |
| experiment_id | str | A/B experiment ID |
| ab_variant | str | Force variant: control \| treatment |

**Response**: `{recommendations: [{isbn, title, authors, ...}]}`
""",
    responses={
        200: {"description": "Personalized recommendations"},
        500: {"description": "Error"},
        503: {"description": "Service not ready"},
    },
)
def personalized_recommendations(
    user_id: str = "local",
    top_k: int = 10,
    limit: Optional[int] = None,
    recent_isbns: Optional[str] = None,
    intent_query: Optional[str] = None,
    experiment_id: Optional[str] = None,
    ab_variant: Optional[str] = None,
):
    """Get personalized recommendations for a user."""
    effective_user_id, k, real_time_seq = parse_request_params(
        user_id, top_k, limit, recent_isbns, intent_query
    )

    # P2: Zero-shot intent probing — when no recent_isbns, use query to seed
    if not real_time_seq and recommender:
        seed = resolve_seed_from_intent(
            intent_query or "", effective_user_id, recommender
        )
        if seed:
            real_time_seq = seed

    if not rec_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    enable_diversity = get_ab_diversity_config(
        effective_user_id, experiment_id, ab_variant
    )

    try:
        recs = rec_service.get_recommendations(
            effective_user_id,
            top_k=k,
            real_time_sequence=real_time_seq,
            enable_diversity_rerank=enable_diversity,
        )
        results = enrich_personal_results(
            recs,
            recommender.vector_db.get_book_details
            if recommender
            else lambda _: {},
        )
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"Error in personalized rec: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/intent/probe",
    summary="Intent probing",
    description="""Zero-shot intent probing for cold-start users. LLM infers categories, emotions, keywords from user's first query.
- **Param**: `query` (str) — e.g. "I want something light and funny"
- **Response**: `{categories: [...], emotions: [...], keywords: [...]}`
""",
    responses={200: {"description": "Inferred intent"}, 500: {"description": "Error"}},
)
def probe_intent_endpoint(query: str = ""):
    """Zero-shot intent probing for cold-start users."""
    from src.core.intent_prober import probe_intent

    try:
        result = probe_intent(query)
        return result
    except Exception as e:
        logger.error(f"Intent probe failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/onboarding/books",
    summary="Onboarding books",
    description="""Return popular books for new-user onboarding. User picks 3–5 to seed preferences (cold-start).
- **Param**: `limit` (int, default 24)
- **Response**: `{books: [{isbn, title, authors, description, thumbnail, category}]}`
""",
    responses={
        200: {"description": "Popular books"},
        500: {"description": "Error"},
        503: {"description": "Service not ready"},
    },
)
def get_onboarding_books(limit: int = 24):
    """Return popular books for new-user onboarding."""
    if not rec_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        items = rec_service.get_popular_books(limit)
        from src.utils import enrich_book_metadata

        results = []
        for isbn, meta in items:
            meta = meta or {}
            meta = enrich_book_metadata(meta, str(isbn))
            results.append(
                {
                    "isbn": isbn,
                    "title": meta.get("title") or f"ISBN: {isbn}",
                    "authors": meta.get("authors", "Unknown"),
                    "description": meta.get("description", ""),
                    "thumbnail": meta.get("thumbnail")
                    or "/content/cover-not-found.jpg",
                    "category": meta.get("category", "General"),
                }
            )
        return {"books": results}
    except Exception as e:
        logger.error(f"Error in onboarding books: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Allow local frontend dev origins
# Added LAST so it wraps the app outermost (first to process request)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Frontend Serving (SPA) ---

# 1. Mount React Assets (JS/CSS)
if os.path.exists("web/dist/assets"):
    app.mount("/assets", StaticFiles(directory="web/dist/assets"), name="assets")

# 2. Mount Local Content Assets (Book Covers)
app.mount("/content", StaticFiles(directory="assets"), name="content")


# 3. Serve React App (Catch-All for Client-Side Routing)
# MUST BE DEFINED LAST to avoid capturing API routes
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # Double check to prevent accidental API capture if regular regex failed
    if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith(
        "openapi"
    ):
        raise HTTPException(status_code=404, detail="Not Found")

    # Serve index.html for all other routes (SPA)
    if os.path.exists("web/dist/index.html"):
        return FileResponse("web/dist/index.html")

    # Fallback if frontend isn't built
    return {
        "message": "Backend is running. Frontend not found (did you run npm build?)",
        "docs_url": "/docs",
    }


__all__ = ["app", "recommender", "rec_service"]

