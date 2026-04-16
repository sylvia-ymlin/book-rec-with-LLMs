"""Admin endpoints: add books, benchmark, and marketing highlights."""
import statistics
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from src.app import state
from src.infra.utils import setup_logger
from src.services.chat_service import chat_service

logger = setup_logger(__name__)

router = APIRouter(tags=["Admin"])


# --- Request models ---

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


class HighlightsRequest(BaseModel):
    """Request for personalized book highlights."""

    isbn: str
    user_id: Optional[str] = Field(default="local")


# --- Endpoints ---

@router.post(
    "/books/add",
    summary="Add book",
    description="Add a new book to the SQLite metadata store and Chroma vector index. ISBN must be unique.",
    responses={
        200: {"description": "Book added successfully"},
        400: {"description": "ISBN already exists or invalid"},
        500: {"description": "Internal error"},
        503: {"description": "Service not ready"},
    },
)
async def add_book_endpoint(req: BookAddRequest):
    if not state.recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        new_book_row = state.recommender.add_new_book(
            req.isbn, req.title, req.author, req.description, req.category, req.thumbnail
        )
        if new_book_row is not None:
            chat_service.add_book_to_context(new_book_row)
            return {"status": "success", "message": f"Book {req.isbn} added."}
        raise HTTPException(status_code=400, detail="Failed to add book. Ensure ISBN is unique.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding book: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/benchmark",
    summary="Performance benchmark",
    description="""Run performance benchmark (5 test queries).
Returns latency stats for vector search and full recommendation pipeline.
- Response: `{vector_search: {mean_ms, median_ms, ...}, full_recommendation: {...}, dataset_size: N}`
""",
    responses={200: {"description": "Benchmark results"}, 503: {"description": "Service not ready"}},
)
async def run_benchmark():
    if not state.recommender:
        raise HTTPException(status_code=503, detail="Service not ready")

    test_queries = [
        "a romantic comedy set in New York",
        "a philosophical novel about the meaning of life",
        "a fast-paced thriller with plot twists",
        "a coming-of-age story about friendship",
        "a science fiction story about space exploration",
    ]

    vector_latencies = []
    for query in test_queries:
        start = time.perf_counter()
        state.recommender.vector_db.search(query, k=50)
        vector_latencies.append((time.perf_counter() - start) * 1000)

    full_latencies = []
    for query in test_queries:
        start = time.perf_counter()
        await state.recommender.get_recommendations(query, "All", "All")
        full_latencies.append((time.perf_counter() - start) * 1000)

    dataset_size = None
    if state.recommender.vector_db.db:
        try:
            dataset_size = state.recommender.vector_db.db._collection.count()
        except Exception:
            pass

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
        "dataset_size": dataset_size,
    }


@router.post(
    "/marketing/highlights",
    summary="Generate book highlights",
    description="Generate a personalized AI highlight for a book using the user's reading persona.",
    responses={200: {"description": "Generated highlights"}, 500: {"description": "Generation failed"}},
)
async def marketing_highlights(req: HighlightsRequest):
    from src.support.marketing.highlights import generate_highlights
    from src.support.marketing.persona import build_persona
    from src.data.stores.profile_store import list_favorites

    try:
        user_id = req.user_id or "local"
        fav_isbns = list_favorites(user_id)  # returns List[str] of ISBNs
        persona = build_persona(fav_isbns)
        return generate_highlights(req.isbn, persona)
    except Exception as e:
        logger.error(f"highlights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
