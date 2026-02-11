from typing import Dict, List, Any
import pandas as pd
import os

from src.utils import setup_logger
from src.core.llm import LLMFactory
from src.user.profile_store import get_cached_highlight, save_cached_highlight

logger = setup_logger(__name__)

# Check for API key in environment (for non-BYOK default mode)
DEFAULT_LLM_KEY = os.getenv("OPENAI_API_KEY", "")


def _first_words(text: str, n: int = 30) -> str:
    try:
        words = str(text).split()
        return (" ".join(words[:n]) + ("..." if len(words) > n else "")).strip()
    except Exception:
        return ""


def generate_highlights(
    isbn: str, 
    persona: Dict[str, Any], 
    books: pd.DataFrame,
    api_key: str = None  # Optional BYOK key
) -> Dict[str, Any]:
    """
    Generate a personalized, LLM-powered highlight about the book.
    Uses persona to tailor the message to the user's reading preferences.
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
    desc = _first_words(str(row.get("description", "")), 50)
    
    # Parse authors
    authors = [a.strip() for a in authors_raw.split(";") if a.strip() and a.strip().lower() != "unknown"]
    author_display = ", ".join(authors) if authors else "Unknown"

    # Extract emotions
    emotions = {
        "joy": float(row.get("joy", 0.0)),
        "sadness": float(row.get("sadness", 0.0)),
        "fear": float(row.get("fear", 0.0)),
        "anger": float(row.get("anger", 0.0)),
        "surprise": float(row.get("surprise", 0.0)),
    }
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"

    persona_summary = persona.get("summary", "a curious reader")

    # --- LLM Generation ---
    # 1. Check Cache First
    # Assuming user_id is passed or available context. For now using 'local' if api_key not set, 
    # but ideally should come from request context. 
    # Since api_key is optional param here, we can infer user context implicitly or explicit param.
    # Note: Using "local" as default user_id for cache to match single-user demo assumption.
    user_id = "local" 
    cached_highlight = get_cached_highlight(user_id, isbn)
    if cached_highlight:
        highlight_text = cached_highlight
    else:
        try:
            # Use local Ollama by default (llama3), fallback to mock if fails
            llm = LLMFactory.create(provider="ollama", model_name="llama3", temperature=0.7)
            
            prompt = f"""You are a literary concierge. Generate a SHORT, personalized highlight (1-2 sentences max) for the following book, tailored to the reader's profile.

Book: "{title}" by {author_display}
Category: {category}
Emotional Tone: {dominant_emotion}
Description: {desc}

Reader Profile: {persona_summary}

Generate a compelling, personalized highlight that explains why THIS reader would enjoy this book. Be concise and engaging. Do not use phrases like "As someone who..." or "Based on your profile...". Just state the value directly."""

            response = llm.invoke(prompt)
            highlight_text = response.content.strip()
            
            # 2. Save to Cache
            save_cached_highlight(user_id, isbn, highlight_text)
            
        except Exception as e:
            logger.warning(f"LLM generation failed, falling back to template: {e}")
            # Fallback to simple template
            highlight_text = f"A {dominant_emotion} {category.lower() if category else 'read'} that resonates with your literary taste."

    return {
        "title": title,
        "authors": author_display,
        "category": category,
        "description": str(row.get("description", "")),
        "highlights": [highlight_text],
        "persona_summary": persona_summary,
        "meta": {
            "title": title,
            "authors": author_display,
            "category": category,
            "description": str(row.get("description", ""))
        }
    }

