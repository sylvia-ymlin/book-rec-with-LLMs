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
