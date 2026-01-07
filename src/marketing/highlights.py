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
    Generate a natural, concise highlight about the book based on its attributes.
    Returns { highlights: List[str], title: str, authors: str, category: str, description: str }
    """
    book_row = books[books["isbn13"].astype(str) == str(isbn)]
    if book_row.empty:
        return {
            "highlights": ["This book brings unique perspectives worth exploring."],
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
    
    # Extract tags and emotions
    tags_raw = str(row.get("tags", ""))
    tags = [t.strip() for t in tags_raw.split(";") if t.strip()][:3]  # top 3 tags
    
    emotions = {
        "joy": float(row.get("joy", 0.0)),
        "sadness": float(row.get("sadness", 0.0)),
        "fear": float(row.get("fear", 0.0)),
        "anger": float(row.get("anger", 0.0)),
        "surprise": float(row.get("surprise", 0.0)),
    }
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
    
    # Build natural language highlight
    parts = []
    
    # Emotional tone
    emotion_map = {
        "joy": "uplifting and heartwarming",
        "sadness": "deeply moving and contemplative",
        "fear": "gripping and suspenseful",
        "anger": "powerful and thought-provoking",
        "surprise": "unexpected and engaging"
    }
    if dominant_emotion and emotions.get(dominant_emotion, 0) > 0.3:
        parts.append(f"A {emotion_map.get(dominant_emotion, 'compelling')} {category.lower() if category else 'read'}")
    elif category:
        parts.append(f"An engaging {category.lower()} work")
    else:
        parts.append("A captivating read")
    
    # Theme tags
    if tags:
        parts.append(f"exploring themes of {', '.join(tags)}")
    
    # Author mention (if available)
    authors = [a.strip() for a in authors_raw.split(";") if a.strip() and a.strip().lower() != "unknown"]
    if authors:
        parts.append(f"by {authors[0]}")
    
    # Construct final sentence
    highlight = " ".join(parts) + "."
    
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
        "highlights": [highlight],
        "persona_summary": persona.get("summary", ""),
        "meta": {
            "title": title,
            "authors": author_display,
            "category": category,
            "description": desc
        }
    }
