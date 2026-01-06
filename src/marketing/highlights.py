from typing import Dict, List, Any
import pandas as pd

from src.utils import setup_logger

logger = setup_logger(__name__)


def _first_words(text: str, n: int = 30) -> str:
    try:
        words = str(text).split()
        return (" ".join(words[:n]) + ("..." if len(words) > n else "")).strip()
    except Exception:
        return ""


def generate_highlights(isbn: str, persona: Dict[str, Any], books: pd.DataFrame) -> Dict[str, Any]:
    """
    Produce concise selling points for a book, optionally tailored by persona.
    Returns { highlights: List[str], title: str, authors: str, category: str, persona_summary: str }
    """
    book_row = books[books["isbn13"].astype(str) == str(isbn)]
    if book_row.empty:
        return {"highlights": ["未找到该书的详细信息。"], "persona_summary": persona.get("summary", ""), "title": "", "authors": "", "category": ""}

    row = book_row.iloc[0]
    title = str(row.get("title", ""))
    authors_raw = str(row.get("authors", ""))
    category = str(row.get("simple_categories", ""))
    desc = str(row.get("description", ""))

    # Persona matches
    matches: List[str] = []
    top_authors = set(persona.get("top_authors", []))
    top_categories = set(persona.get("top_categories", []))
    authors = [a.strip() for a in authors_raw.split(";") if a.strip()]

    if any(a in top_authors for a in authors):
        matches.append("契合你的偏好作者")
    if category and category in top_categories:
        matches.append("符合你常看类别")

    desc_snippet = _first_words(desc, 30)
    bullets: List[str] = []
    if matches:
        bullets.append("、".join(matches) + "：更对你的口味")
    if authors:
        bullets.append(f"作者：{', '.join(authors[:3])}")
    if category:
        bullets.append(f"类别：{category}")
    if desc_snippet:
        bullets.append(f"简介摘要：{desc_snippet}")
    bullets.append("适合：对该主题感兴趣、偏好同类作品的读者")

    return {
        "title": title,
        "authors": ", ".join(authors) if authors else authors_raw,
        "category": category,
        "highlights": bullets[:5],
        "persona_summary": persona.get("summary", ""),
    }
