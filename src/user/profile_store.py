import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from src.utils import setup_logger
from src.config import DATA_DIR

logger = setup_logger(__name__)

STORE_PATH = DATA_DIR / "user_profiles.json"

# Reading status enum values
READING_STATUS = ["want_to_read", "reading", "finished"]


def _ensure_store_file() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not STORE_PATH.exists():
            with open(STORE_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f)
    except Exception as e:
        logger.error(f"Failed to ensure profile store: {e}")
        raise


def _load_store() -> Dict[str, Dict[str, Any]]:
    _ensure_store_file()
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except json.JSONDecodeError:
            return {}


def _save_store(store: Dict[str, Dict[str, Any]]) -> None:
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _migrate_favorites(user: Dict) -> Dict:
    """Migrate old format (list of ISBNs) to new format (dict with metadata)."""
    if "favorites" not in user:
        user["favorites"] = {}
        return user
    
    if isinstance(user["favorites"], list):
        # Old format: ["isbn1", "isbn2"]
        old_list = user["favorites"]
        new_dict = {}
        for isbn in old_list:
            new_dict[str(isbn)] = {
                "added_at": datetime.now().isoformat(),
                "rating": None,
                "status": "want_to_read"
            }
        user["favorites"] = new_dict
    
    return user


def add_favorite(user_id: str, isbn: str) -> int:
    """Add an ISBN to user's favorites. Returns new favorites count."""
    store = _load_store()
    user = store.get(user_id) or {"favorites": {}}
    user = _migrate_favorites(user)
    
    favorites = user.get("favorites", {})
    isbn_str = str(isbn)
    
    if isbn_str not in favorites:
        favorites[isbn_str] = {
            "added_at": datetime.now().isoformat(),
            "rating": None,
            "status": "want_to_read"
        }
    
    user["favorites"] = favorites
    store[user_id] = user
    _save_store(store)
    return len(favorites)


def remove_favorite(user_id: str, isbn: str) -> int:
    """Remove an ISBN from user's favorites. Returns new favorites count."""
    store = _load_store()
    user = store.get(user_id) or {"favorites": {}}
    user = _migrate_favorites(user)
    
    favorites = user.get("favorites", {})
    isbn_str = str(isbn)
    
    if isbn_str in favorites:
        del favorites[isbn_str]
    
    user["favorites"] = favorites
    store[user_id] = user
    _save_store(store)
    return len(favorites)


def update_book_rating(user_id: str, isbn: str, rating: float) -> bool:
    """Update the user's rating for a book. Rating should be 1-5."""
    if rating < 1 or rating > 5:
        return False
    
    store = _load_store()
    user = store.get(user_id) or {"favorites": {}}
    user = _migrate_favorites(user)
    
    favorites = user.get("favorites", {})
    isbn_str = str(isbn)
    
    if isbn_str not in favorites:
        # Auto-add to favorites if rating
        favorites[isbn_str] = {
            "added_at": datetime.now().isoformat(),
            "rating": rating,
            "status": "want_to_read"
        }
    else:
        favorites[isbn_str]["rating"] = rating
    
    user["favorites"] = favorites
    store[user_id] = user
    _save_store(store)
    return True


def update_reading_status(user_id: str, isbn: str, status: str) -> bool:
    """Update the reading status for a book."""
    if status not in READING_STATUS:
        return False
    
    store = _load_store()
    user = store.get(user_id) or {"favorites": {}}
    user = _migrate_favorites(user)
    
    favorites = user.get("favorites", {})
    isbn_str = str(isbn)
    
    if isbn_str not in favorites:
        favorites[isbn_str] = {
            "added_at": datetime.now().isoformat(),
            "rating": None,
            "status": status,
            "comment": ""
        }
    else:
        favorites[isbn_str]["status"] = status
        if status == "finished" and "finished_at" not in favorites[isbn_str]:
            favorites[isbn_str]["finished_at"] = datetime.now().isoformat()
    
    user["favorites"] = favorites
    store[user_id] = user
    _save_store(store)
    return True

def update_book_comment(user_id: str, isbn: str, comment: str) -> bool:
    store = _load_store()
    user = store.get(user_id) or {"favorites": {}}
    user = _migrate_favorites(user)
    
    favorites = user.get("favorites", {})
    isbn_str = str(isbn)
    
    if isbn_str not in favorites:
        favorites[isbn_str] = {
            "added_at": datetime.now().isoformat(),
            "rating": None,
            "status": "want_to_read",
            "comment": comment
        }
    else:
        favorites[isbn_str]["comment"] = comment
    
    user["favorites"] = favorites
    store[user_id] = user
    _save_store(store)
    return True


def list_favorites(user_id: str) -> List[str]:
    """Return list of favorite ISBNs (for backward compatibility)."""
    store = _load_store()
    user = store.get(user_id) or {"favorites": {}}
    user = _migrate_favorites(user)
    
    favorites = user.get("favorites", {})
    return list(favorites.keys())


def get_favorites_with_metadata(user_id: str) -> Dict[str, Dict]:
    """Return favorites with full metadata (rating, status, timestamps)."""
    store = _load_store()
    user = store.get(user_id) or {"favorites": {}}
    user = _migrate_favorites(user)
    
    return user.get("favorites", {})


def get_reading_stats(user_id: str) -> Dict[str, int]:
    """Return reading statistics for user."""
    favorites = get_favorites_with_metadata(user_id)
    
    stats = {
        "total": len(favorites),
        "want_to_read": 0,
        "reading": 0,
        "finished": 0,
        "rated": 0
    }
    
    for isbn, meta in favorites.items():
        status = meta.get("status", "want_to_read")
        if status in stats:
            stats[status] += 1
        if meta.get("rating") is not None:
            stats["rated"] += 1
    
    return stats


def save_cached_highlight(user_id: str, isbn: str, highlight: str) -> None:
    """Save a generated highlight to the user's profile."""
    store = _load_store()
    user = store.get(user_id) or {"favorites": {}, "cached_highlights": {}}
    
    if "cached_highlights" not in user:
        user["cached_highlights"] = {}
    
    user["cached_highlights"][str(isbn)] = highlight
    store[user_id] = user
    _save_store(store)


def get_cached_highlight(user_id: str, isbn: str) -> str | None:
    """Retrieve a cached highlight if it exists."""
    store = _load_store()
    user = store.get(user_id) or {}
    return user.get("cached_highlights", {}).get(str(isbn))
