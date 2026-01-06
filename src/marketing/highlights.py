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
    Returns { highlights: List[str], title: str, authors: str, category: str, description: str }
    """
    book_row = books[books["isbn13"].astype(str) == str(isbn)]
    if book_row.empty:
        return {
            "highlights": ["Book details not found in our collection."],
            "persona_summary": persona.get("summary", ""),
            "title": "",
            "authors": "Unknown",
            "category": "",
            "description": ""
        }

    row = book_row.iloc[0]
    title = str(row.get("title", ""))
    authors_raw = str(row.get("authors", ""))
    category = str(row.get("simple_categories", ""))
    desc = str(row.get("description", ""))

    # Persona matches
    matches: List[str] = []
    top_authors = set(persona.get("top_authors", []))
    top_categories = set(persona.get("top_categories", []))
    authors = [a.strip() for a in authors_raw.split(";") if a.strip() and a.strip().lower() != "unknown"]

    if any(a in top_authors for a in authors):
        matches.append("Matches your favorite authors")
    if category and category in top_categories:
        matches.append("Aligns with your reading preferences")

    desc_snippet = _first_words(desc, 30)
    bullets: List[str] = []
    
    if matches:
        bullets.append(" • ".join(matches))
    
    if authors:
        bullets.append(f"By: {', '.join(authors[:3])}")
    
    if category:
        bullets.append(f"Category: {category}")
    
    if desc_snippet:
        bullets.append(f"Summary: {desc_snippet}")
    
    bullets.append("Perfect for readers who love this genre and thematic exploration")

    # Handle author display
    if authors and authors_raw.lower() != "unknown":
        author_display = ", ".join(authors)
    else:
        author_display = "Unknown"

    return {
        "title": title,
        "authors": author_display,
        "category": category,
        "description": desc,
        "highlights": bullets[:5],
        "persona_summary": persona.get("summary", ""),
        "meta": {
            "title": title,
            "authors": author_display,
            "category": category,
            "description": desc
        }
    }
