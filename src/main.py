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
    version="2.6.0"
)

# Include Routers
app.include_router(chat_router)



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
    
    # Download models from HF Hub if not present (for HF Spaces deployment)
    from src.core.model_loader import ensure_models_exist
    logger.info("Checking/downloading models from HF Hub...")
    ensure_models_exist()
    
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
    user_id: Optional[str] = "local"
    use_agentic: Optional[bool] = False  # LangGraph workflow: Router -> Retrieve -> Evaluate -> Web Fallback


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str  # "positive" or "negative"

class BookResponse(BaseModel):
    isbn: str
    title: str
    authors: str
    description: str
    thumbnail: str
    caption: str
    tags: List[str] = []
    average_rating: float = 0.0
    explanations: List[FeatureContribution] = []  # SHAP explanations (V2.7)


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
    Set use_agentic: true for LangGraph workflow (Router -> Retrieve -> Evaluate -> Web Fallback).
    Async to avoid blocking event loop (web search fallback uses httpx).
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        results = await recommender.get_recommendations(
            query=request.query,
            category=request.category,
            user_id=request.user_id if hasattr(request, 'user_id') else "local",
            use_agentic=request.use_agentic or False,
        )
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend/similar/{isbn}", response_model=RecommendationResponse)
def get_similar_books(isbn: str, k: int = 10, category: str = "All"):
    """
    Content-based similar books by vector similarity.
    
    When user clicks a book, call this to show similar recommendations immediately.
    No user history required; works for new users and new books in ChromaDB.
    """
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        results = recommender.get_similar_books(isbn=isbn, k=k, category=category)
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"get_similar_books error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
async def get_categories():
    if not recommender:
         raise HTTPException(status_code=503, detail="Service not ready")
    return {"categories": recommender.get_categories()}




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


@app.delete("/favorites/remove")
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


@app.get("/favorites/list/{user_id}")
async def favorites_list(user_id: str):
    """Return user's favorite books with full details."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        # Get favorites with metadata (rating, status)
        favorites_meta = get_favorites_with_metadata(user_id)
        # ENGINEERING IMPROVEMENT: Zero-RAM Lookup
        from src.core.metadata_store import metadata_store
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

            results.append({
                "isbn": isbn,
                "title": title,
                "author": author,
                "img": thumbnail,
                "category": book_meta.get("simple_categories", ""),
                "rating": meta.get("rating"),
                "status": meta.get("status", "want_to_read"),
                "added_at": meta.get("added_at"),
                "comment": meta.get("comment", "")
            })
        return {"favorites": results}
    except Exception as e:
        logger.error(f"favorites_list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ... (intervening code) ...

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


# --- Personalized Recommendation API ---

@app.get("/api/recommend/personal", response_model=RecommendationResponse)
def personalized_recommendations(user_id: str = "local", top_k: int = 10):
    """
    Get personalized recommendations for a user.
    Uses 6-channel recall (ItemCF/UserCF/Swing/SASRec/YoutubeDNN/Popularity) + LGBMRanker.
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
        from src.utils import enrich_book_metadata
        
        results = []
        for isbn, score, explanation in recs:
            # Recommender matches our singleton 'recommender'
            meta = recommender.vector_db.get_book_details(isbn) or {}
            
            # Enrich with dynamic cover fetching
            meta = enrich_book_metadata(meta, str(isbn))
            
            # Fallback for display
            title = meta.get("title") or f"ISBN: {isbn}"
            desc = meta.get("description", "No description available.")
            thumb = meta.get("thumbnail", "/content/cover-not-found.jpg")
            authors = meta.get("authors", "Unknown")
            
            # More robust rating/metadata mapping
            rating = 0.0
            if meta:
                # Try average_rating or rating
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
            
            # Format cover
            if not thumb:
                 thumb = "/content/cover-not-found.jpg"
            
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
                "explanations": explanation,  # SHAP feature contributions (V2.7)
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

# --- Frontend Serving (SPA) ---
import os
from fastapi.responses import FileResponse

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
    if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi"):
        raise HTTPException(status_code=404, detail="Not Found")
        
    # Serve index.html for all other routes (SPA)
    if os.path.exists("web/dist/index.html"):
        return FileResponse("web/dist/index.html")
    
    # Fallback if frontend isn't built
    return {
        "message": "Backend is running. Frontend not found (did you run npm build?)",
        "docs_url": "/docs"
    }
