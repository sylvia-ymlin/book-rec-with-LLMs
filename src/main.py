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
from src.user.profile_store import add_favorite, list_favorites
from src.marketing.persona import build_persona
from src.marketing.highlights import generate_highlights
from src.api.chat import router as chat_router # ✨ NEW

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

# Allow local frontend dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Initialize Recommender (Singleton)
# We do this on startup so the first request is fast
recommender = None

@app.on_event("startup")
async def startup_event():
    global recommender
    logger.info("Initializing Recommender Engine...")
    recommender = BookRecommender()
    logger.info("Recommender Engine Initialized.")

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

