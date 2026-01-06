from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.recommender import BookRecommender
from src.utils import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Book Recommender API",
    description="API for Intelligent Book Recommendation System",
    version="1.0.0"
)

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

class BookResponse(BaseModel):
    isbn: str
    title: str
    authors: str
    description: str
    thumbnail: str
    caption: str

class RecommendationResponse(BaseModel):
    recommendations: List[BookResponse]

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
            tone=request.tone
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

