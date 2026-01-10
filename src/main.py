from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import prometheus_client
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.recommender import BookRecommender
from src.utils import setup_logger
from src.user.profile_store import (
    add_favorite, list_favorites, remove_favorite,
    update_book_rating, update_reading_status, update_book_comment,
    get_favorites_with_metadata, get_reading_stats
)
from src.marketing.persona import build_persona
from src.marketing.highlights import generate_highlights
from src.api.chat import router as chat_router # ✨ NEW
from src.services.chat_service import chat_service # ✨ NEW
from src.services.recommend_service import RecommendationService # ✨ NEW

logger = setup_logger(__name__)

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter("http_requests_total", "Total count of HTTP requests", ["method", "endpoint", "status_code"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency in seconds", ["method", "endpoint"])


app = FastAPI(
    title="Book Recommender API",
    description="API for Intelligent Book Recommendation System (RAG Capabilities Enabled)",
    version="2.0.0"
)

# Include Routers
app.include_router(chat_router)

# 挂载静态目录，确保前端能访问 /assets/cover-not-found.jpg
app.mount("/assets", StaticFiles(directory="assets"), name="assets")



# --- Observability Middleware ---
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
    except Exception as e:
        status = "500"
        raise e
    finally:
        process_time = time.perf_counter() - start_time
        REQUEST_COUNT.labels(method=method, endpoint=path, status_code=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(process_time)
        
    return response

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Initialize Recommender and Services (Singleton)
# We do this on startup so the first request is fast
recommender = None
rec_service = None # ✨ NEW

@app.on_event("startup")
async def startup_event():
    global recommender, rec_service
    logger.info("Initializing Recommender Engine...")
    recommender = BookRecommender()
    
    logger.info("Initializing Personalized Rec Service...")
    rec_service = RecommendationService()
    # Pre-warm resources for better UX
    try:
        rec_service.load_resources()
    except Exception as e:
        logger.error(f"Failed to pre-load resources: {e}")
    
    logger.info("Engines Initialized.")

# Pydantic Models
class RecommendationRequest(BaseModel):
    query: str
    category: str = "All"
    tone: str = "All"
    user_id: Optional[str] = "local"

class BookResponse(BaseModel):
    isbn: str
    title: str
    authors: str
    description: str
    thumbnail: str
    caption: str
    tags: List[str] = []
    emotions: Dict[str, float] = {}
    review_highlights: List[str] = []
    average_rating: float = 0.0

class RecommendationResponse(BaseModel):
    recommendations: List[BookResponse]


class FavoriteRequest(BaseModel):
    user_id: Optional[str] = "local"
    isbn: str


class HighlightsRequest(BaseModel):
    isbn: str
    user_id: Optional[str] = "local"


class BookUpdateRequest(BaseModel):
    user_id: Optional[str] = "local"
    isbn: str
    rating: Optional[float] = None
    status: Optional[str] = None  # "want_to_read", "reading", "finished"
    comment: Optional[str] = None

class BookAddRequest(BaseModel):
    isbn: str
    title: str
    author: str
    description: str
    category: Optional[str] = "General"
    thumbnail: Optional[str] = None

@app.post("/books/add")
async def add_book_endpoint(req: BookAddRequest):
    """
    Dynamically add a new book to the database and vector index.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        new_book_row = recommender.add_new_book(req.isbn, req.title, req.author, req.description, req.category, req.thumbnail)
        if new_book_row is not None:
             # Also update ChatService context
            chat_service.add_book_to_context(new_book_row)
            return {"status": "success", "message": f"Book {req.isbn} added."}
        else:
            raise HTTPException(status_code=400, detail="Failed to add book. Ensure ISBN is unique.")
    except Exception as e:
        logger.error(f"Error adding book: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Generate book recommendations based on semantic search and emotion/category filtering.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        results = recommender.get_recommendations(
            query=request.query,
            category=request.category,
            tone=request.tone,
            user_id=request.user_id if hasattr(request, 'user_id') else "local"
        )
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    if not recommender:
         raise HTTPException(status_code=503, detail="Service not ready")
    return {"categories": recommender.get_categories()}

@app.get("/tones")
async def get_tones():
    if not recommender:
         raise HTTPException(status_code=503, detail="Service not ready")
    return {"tones": recommender.get_tones()}


# --- Favorites & Persona & Highlights ---
@app.post("/favorites/add")
async def favorites_add(req: FavoriteRequest):
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        count = add_favorite(req.user_id or "local", req.isbn)
        return {"status": "ok", "favorites_count": count}
    except Exception as e:
        logger.error(f"favorites_add error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/favorites/list/{user_id}")
async def favorites_list(user_id: str):
    """Return user's favorite books with full details."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        # Get favorites with metadata (rating, status)
        favorites_meta = get_favorites_with_metadata(user_id)
        books_df = recommender.books
        results = []
        for isbn, meta in favorites_meta.items():
            book_row = books_df[books_df["isbn13"].astype(str) == str(isbn)]
            if not book_row.empty:
                row = book_row.iloc[0]
                results.append({
                    "isbn": isbn,
                    "title": row.get("title", ""),
                    "author": row.get("authors", "Unknown"),
                    "img": row.get("thumbnail", "/assets/cover-not-found.jpg"),
                    "category": row.get("simple_categories", ""),
                    "mood": "Joy" if row.get("joy", 0) > 0.3 else "Neutral",
                    "rating": meta.get("rating"),
                    "status": meta.get("status", "want_to_read"),
                    "added_at": meta.get("added_at"),
                    "comment": meta.get("comment", "")
                })
        return {"favorites": results}
    except Exception as e:
        logger.error(f"favorites_list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/favorites/update")
async def favorites_update(req: BookUpdateRequest):
    """Update rating or reading status for a book."""
    try:
        user_id = req.user_id or "local"
        results = {}
        
        if req.rating is not None:
            success = update_book_rating(user_id, req.isbn, req.rating)
            results["rating_updated"] = success
        
        if req.status is not None:
            success = update_reading_status(user_id, req.isbn, req.status)
            results["status_updated"] = success
            
        if req.comment is not None:
            success = update_book_comment(user_id, req.isbn, req.comment)
            results["comment_updated"] = success
        
        return {"status": "ok", **results}
    except Exception as e:
        logger.error(f"favorites_update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/favorites/remove")
async def favorites_remove(req: FavoriteRequest):
    """Remove a book from favorites."""
    try:
        count = remove_favorite(req.user_id or "local", req.isbn)
        return {"status": "ok", "favorites_count": count}
    except Exception as e:
        logger.error(f"favorites_remove error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/stats")
async def user_stats(user_id: str):
    """Get reading statistics for a user."""
    try:
        stats = get_reading_stats(user_id)
        return stats
    except Exception as e:
        logger.error(f"user_stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/persona")
async def user_persona(user_id: str):
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        favs = list_favorites(user_id)
        persona = build_persona(favs, recommender.books)
        return {"user_id": user_id, "favorites": favs, "persona": persona}
    except Exception as e:
        logger.error(f"user_persona error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/marketing/highlights")
async def marketing_highlights(req: HighlightsRequest):
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        favs = list_favorites(req.user_id or "local")
        persona = build_persona(favs, recommender.books)
        result = generate_highlights(req.isbn, persona, recommender.books)
        # highlights和meta.description都unescape
        from html import unescape
        highlights = [unescape(h) for h in result.get("highlights", [])]
        meta = result.copy()
        if "description" in meta:
            meta["description"] = unescape(meta["description"])
        return {"persona": persona, "highlights": highlights, "meta": meta}
    except Exception as e:
        logger.error(f"marketing_highlights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmark")
async def run_benchmark():
    """
    Run performance benchmark and return latency metrics.
    Tests vector search and full recommendation pipeline.
    """
    import time
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
    
    # Benchmark full recommendation
    full_latencies = []
    for query in test_queries:
        start = time.perf_counter()
        recommender.get_recommendations(query, "All", "All")
        full_latencies.append((time.perf_counter() - start) * 1000)
    
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
        "dataset_size": len(recommender.books),
    }


# --- Personalized Recommendation API ---

@app.get("/api/recommend/personal")
async def personalized_recommendations(user_id: str = "local", top_k: int = 10):
    """
    Get personalized recommendations for a user.
    Uses multi-channel recall (ItemCF/UserCF) + XGBoost Ranking.
    """
    # Demo logic: Map 'local' to a real user for demonstration
    if user_id in ["local", "demo"]:
        # Pick a demo user ID from active users (A1ZQ1LUQ9R6JHZ is a heavy reader)
        user_id = "A1ZQ1LUQ9R6JHZ" 
    
    # Check initialization
    if not rec_service:
        raise HTTPException(status_code=503, detail="Service not ready")
        
    try:
        recs = rec_service.get_recommendations(user_id, top_k)
        
        # Enrich with metadata
        results = []
        for isbn, score in recs:
            # Recommender matches our singleton 'recommender'
            meta = recommender.vector_db.get_book_details(isbn)
            
            # If meta missing, fallback to partial info
            title = meta.get("title", f"ISBN: {isbn}") if meta else f"ISBN: {isbn}"
            desc = meta.get("description", "No description available.") if meta else ""
            thumb = meta.get("thumbnail", "") if meta else ""
            author = meta.get("authors", "") if meta else ""
            
            # Format cover
            if not thumb:
                 thumb = "/assets/cover-not-found.jpg"
            
            results.append({
                "isbn": isbn,
                "score": float(score),
                "title": title,
                "author": author,
                "description": desc,
                "thumbnail": thumb
            })
            
        return {"recommendations": results}
        
    except Exception as e:
        logger.error(f"Error in personalized rec: {e}")
        # In production, maybe return fallback popular items instead of error
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
