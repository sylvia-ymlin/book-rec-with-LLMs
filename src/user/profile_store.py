import json
import os
from typing import Dict, List
from pathlib import Path

from src.utils import setup_logger
from src.config import DATA_DIR

logger = setup_logger(__name__)

STORE_PATH = DATA_DIR / "user_profiles.json"


def _ensure_store_file() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not STORE_PATH.exists():
            with open(STORE_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f)
    except Exception as e:
        logger.error(f"Failed to ensure profile store: {e}")
        raise


def _load_store() -> Dict[str, Dict[str, List[str]]]:
    _ensure_store_file()
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except json.JSONDecodeError:
            return {}


def _save_store(store: Dict[str, Dict[str, List[str]]]) -> None:
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def add_favorite(user_id: str, isbn: str) -> int:
    """Add an ISBN to user's favorites. Returns new favorites count."""
    store = _load_store()
    user = store.get(user_id) or {"favorites": []}
    favs: List[str] = list(dict.fromkeys(user.get("favorites", [])))
    if isbn not in favs:
        favs.append(isbn)
    user["favorites"] = favs
    store[user_id] = user
    _save_store(store)
    return len(favs)


def list_favorites(user_id: str) -> List[str]:
    store = _load_store()
    user = store.get(user_id) or {"favorites": []}
    favs: List[str] = user.get("favorites", [])
    # De-duplicate while preserving order
    return list(dict.fromkeys(favs))


def save_cached_highlight(user_id: str, isbn: str, highlight: str) -> None:
    """Save a generated highlight to the user's profile."""
    store = _load_store()
    user = store.get(user_id) or {"favorites": [], "cached_highlights": {}}
    
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

