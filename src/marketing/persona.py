from collections import Counter
from typing import Dict, List, Any
import pandas as pd

from src.utils import setup_logger

logger = setup_logger(__name__)


def build_persona(fav_isbns: List[str], books: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate a simple persona from favorites: top authors and categories."""
    if not isinstance(books, pd.DataFrame) or books.empty or not fav_isbns:
        return {
            "summary": "暂无画像。先收藏几本你喜欢的书，获得更个性化的卖点。",
            "top_authors": [],
            "top_categories": [],
        }

    fav_df = books[books["isbn13"].astype(str).isin([str(x) for x in fav_isbns])]
    authors_list: List[str] = []
    categories_list: List[str] = []

    for _, row in fav_df.iterrows():
        # Authors are ';' separated in our dataset
        try:
            authors_list.extend([a.strip() for a in str(row.get("authors", "")).split(";") if a.strip()])
        except Exception:
            pass
        cat = str(row.get("simple_categories", "")).strip()
        if cat:
            categories_list.append(cat)

    top_authors = [a for a, _ in Counter(authors_list).most_common(3)]
    top_categories = [c for c, _ in Counter(categories_list).most_common(3)]

    if not top_authors and not top_categories:
        return {
            "summary": "画像尚不明确，继续收藏以完善偏好画像。",
            "top_authors": [],
            "top_categories": [],
        }

    parts: List[str] = []
    if top_authors:
        parts.append(f"偏爱作者：{', '.join(top_authors)}")
    if top_categories:
        parts.append(f"常看类别：{', '.join(top_categories)}")

    return {
        "summary": "；".join(parts),
        "top_authors": top_authors,
        "top_categories": top_categories,
    }
