from collections import Counter
from typing import Dict, List, Any
import pandas as pd

from src.utils import setup_logger

logger = setup_logger(__name__)


def build_persona(fav_isbns: List[str], books: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate a simple persona from favorites: top authors and categories."""
    if not isinstance(books, pd.DataFrame) or books.empty or not fav_isbns:
        return {
            "summary": "No profile yet. Start by adding your favorite books to see personalized recommendations.",
            "top_authors": [],
            "top_categories": [],
        }

    # ENGINEERING IMPROVEMENT: Zero-RAM Lookup
    from src.core.metadata_store import metadata_store
    
    authors_list: List[str] = []
    categories_list: List[str] = []

    for isbn in fav_isbns:
        meta = metadata_store.get_book_metadata(str(isbn))
        if not meta:
            continue
            
        # Authors
        try:
            author_str = str(meta.get("authors", "")).strip()
            if author_str and author_str.lower() != "unknown":
                authors_list.extend([a.strip() for a in author_str.split(";") if a.strip() and a.strip().lower() != "unknown"])
        except Exception:
            pass
            
        # Categories
        cat = str(meta.get("simple_categories", "")).strip()
        if cat and cat.lower() != "unknown":
            categories_list.append(cat)

    top_authors = [a for a, _ in Counter(authors_list).most_common(3)]
    top_categories = [c for c, _ in Counter(categories_list).most_common(3)]

    if not top_authors and not top_categories:
        return {
            "summary": "Your profile is taking shape. Keep adding books to refine your taste profile.",
            "top_authors": [],
            "top_categories": [],
        }

    parts: List[str] = []
    if top_authors:
        parts.append(f"You love authors: {', '.join(top_authors)}")
    if top_categories:
        parts.append(f"You often read: {', '.join(top_categories)}")

    return {
        "summary": " | ".join(parts),
        "top_authors": top_authors,
        "top_categories": top_categories,
    }
