"""Bookshelf endpoints: favorites management, categories, and onboarding."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from src.app import state
from src.infra.utils import setup_logger, enrich_book_metadata
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

logger = setup_logger(__name__)

router = APIRouter(tags=["Bookshelf"])


# --- Request models ---

class FavoriteRequest(BaseModel):
    """Request for add/remove favorite."""

    user_id: Optional[str] = Field(default="local")
    isbn: str = Field(..., description="ISBN of the book")


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


# --- Endpoints ---

@router.get(
    "/categories",
    summary="List categories",
    description="Return all book categories available for filtering.",
    responses={200: {"description": "List of category names"}, 503: {"description": "Service not ready"}},
)
async def get_categories():
    if not state.recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"categories": state.recommender.get_categories()}


@router.post(
    "/favorites/add",
    summary="Add favorite",
    description="Add a book to user's favorites. Response: `{status: 'ok', favorites_count: N}`.",
    responses={200: {"description": "Added"}, 500: {"description": "Error"}, 503: {"description": "Service not ready"}},
)
async def favorites_add(req: FavoriteRequest):
    if not state.recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        count = add_favorite(req.user_id or "local", req.isbn)
        return {"status": "ok", "favorites_count": count}
    except Exception as e:
        logger.error(f"favorites_add error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/favorites/remove",
    summary="Remove favorite",
    description="Remove a book from user's favorites. Response: `{status: 'ok', favorites_count: N}`.",
    responses={200: {"description": "Removed"}, 500: {"description": "Error"}, 503: {"description": "Service not ready"}},
)
async def favorites_remove(req: FavoriteRequest):
    if not state.recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        count = remove_favorite(req.user_id or "local", req.isbn)
        return {"status": "ok", "favorites_count": count}
    except Exception as e:
        logger.error(f"favorites_remove error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/favorites/update",
    summary="Update favorite",
    description="""Update rating, reading status, or comment for a favorited book.
- **rating**: 1–5 (optional)
- **status**: `want_to_read` | `reading` | `finished` (optional)
- **comment**: Free text (optional)
All fields optional; only provided fields are updated.
""",
    responses={200: {"description": "Updated"}, 500: {"description": "Error"}, 503: {"description": "Service not ready"}},
)
async def favorites_update(req: BookUpdateRequest):
    if not state.recommender:
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


@router.get(
    "/favorites/list/{user_id}",
    summary="List favorites",
    description="""Return user's favorite books with full metadata.
- Response: `{favorites: [{isbn, title, author, img, category, rating, status, added_at, comment}]}`
""",
    responses={200: {"description": "List of favorites"}, 500: {"description": "Error"}, 503: {"description": "Service not ready"}},
)
async def favorites_list(user_id: str):
    if not state.recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        from src.data.stores.metadata_store import metadata_store

        favorites_meta = get_favorites_with_metadata(user_id)
        results = []
        for isbn, meta in favorites_meta.items():
            book_meta = metadata_store.get_book_metadata(str(isbn))
            book_meta = enrich_book_metadata(book_meta, str(isbn))
            results.append(
                {
                    "isbn": isbn,
                    "title": book_meta.get("title") or f"Unknown Book ({isbn})",
                    "author": book_meta.get("authors", "Unknown"),
                    "img": book_meta.get("thumbnail") or "/content/cover-not-found.jpg",
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


@router.get(
    "/user/{user_id}/stats",
    summary="User reading stats",
    description="Return reading statistics for a user: total favorites, want_to_read, reading, finished, rated.",
    responses={200: {"description": "Reading stats"}, 500: {"description": "Error"}},
)
async def user_stats(user_id: str):
    try:
        return get_reading_stats(user_id)
    except Exception as e:
        logger.error(f"user_stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/user/{user_id}/persona",
    summary="User persona",
    description="Return aggregated persona from user's favorites: top authors, top categories, summary.",
    responses={200: {"description": "Persona"}, 500: {"description": "Error"}},
)
async def user_persona(user_id: str):
    from src.support.marketing.persona import build_persona

    try:
        fav_isbns = list_favorites(user_id)
        return build_persona(fav_isbns)
    except Exception as e:
        logger.error(f"user_persona error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/onboarding/books",
    summary="Onboarding books",
    description="""Return popular books for new-user onboarding.
User picks 3–5 to seed preferences (cold-start).
- Param: `limit` (int, default 24)
- Response: `{books: [{isbn, title, authors, description, thumbnail, category}]}`
""",
    responses={200: {"description": "Popular books"}, 500: {"description": "Error"}, 503: {"description": "Service not ready"}},
)
def get_onboarding_books(limit: int = 24):
    if not state.rec_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        items = state.rec_service.get_popular_books(limit)
        results = []
        for isbn, meta in items:
            meta = enrich_book_metadata(meta or {}, str(isbn))
            results.append(
                {
                    "isbn": isbn,
                    "title": meta.get("title") or f"ISBN: {isbn}",
                    "authors": meta.get("authors", "Unknown"),
                    "description": meta.get("description", ""),
                    "thumbnail": meta.get("thumbnail") or "/content/cover-not-found.jpg",
                    "category": meta.get("category", "General"),
                }
            )
        return {"books": results}
    except Exception as e:
        logger.error(f"Error in onboarding books: {e}")
        raise HTTPException(status_code=500, detail=str(e))
