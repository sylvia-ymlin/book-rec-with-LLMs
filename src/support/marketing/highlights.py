from typing import Dict, List, Any
import pandas as pd

from src.infra.utils import setup_logger
from src.rag.llm import LLMFactory
from src.data.stores.profile_store import get_cached_highlight, save_cached_highlight
from src.data.stores.metadata_store import metadata_store

logger = setup_logger(__name__)


def _first_words(text: str, n: int = 30) -> str:
    try:
        words = str(text).split()
        return (" ".join(words[:n]) + ("..." if len(words) > n else "")).strip()
    except Exception:
        return ""


def generate_highlights(
    isbn: str,
    persona: Dict[str, Any],
    books: pd.DataFrame = None,  # Deprecated, unused
    provider: str = "ollama",
    api_key: str | None = None,
) -> Dict[str, Any]:
    """Generate a personalized highlight for a book using LLM."""

    # Fetch book metadata from store
    meta = metadata_store.get_book_metadata(isbn) or {}
    title = meta.get("title", f"Book {isbn}")
    authors_raw = meta.get("authors", "Unknown")
    author_display = authors_raw if isinstance(authors_raw, str) else ", ".join(authors_raw)
    category = meta.get("category", "General")
    desc = _first_words(str(meta.get("description", "")), 50)

    # Parse emotions (stored as dict or JSON string)
    emotions = {}
    raw_emotions = meta.get("emotions", {})
    if isinstance(raw_emotions, dict):
        emotions = raw_emotions
    elif isinstance(raw_emotions, str):
        try:
            import json
            emotions = json.loads(raw_emotions)
        except Exception:
            pass

    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"

    persona_summary = persona.get("summary", "a curious reader")

    # --- LLM Generation ---
    # 1. Check Cache First
    user_id = "local"
    cached_highlight = get_cached_highlight(user_id, isbn)
    if cached_highlight:
        highlight_text = cached_highlight
    else:
        try:
            model_name = None
            if provider == "groq":
                model_name = "llama3-70b-8192"
            elif provider == "openai":
                model_name = "gpt-3.5-turbo"
            elif provider == "deepseek":
                model_name = "deepseek-chat"
            elif provider == "ollama":
                model_name = "llama3"

            if provider in {"openai", "groq", "deepseek"} and not api_key:
                raise ValueError(f"API key is required for provider '{provider}'")

            llm = LLMFactory.create(provider=provider, api_key=api_key, model_name=model_name, temperature=0.7)

            prompt = f"""You are a literary concierge. Generate a SHORT, personalized highlight (1-2 sentences max) for the following book, tailored to the reader's profile.

Book: "{title}" by {author_display}
Category: {category}
Emotional Tone: {dominant_emotion}
Description: {desc}

Reader Profile: {persona_summary}

Generate a compelling, personalized highlight that explains why THIS reader would enjoy this book. Be concise and engaging. Do not use phrases like "As someone who..." or "Based on your profile...". Just state the value directly."""

            response = llm.invoke(prompt)
            if isinstance(response, str):
                highlight_text = response.strip()
            else:
                highlight_text = response.content.strip()

            # Save to Cache
            save_cached_highlight(user_id, isbn, highlight_text)

        except Exception as e:
            logger.warning(f"LLM generation failed, falling back to template: {e}")
            # Fallback to simple template
            highlight_text = f"A {dominant_emotion} {category.lower() if category else 'read'} that resonates with your literary taste."

    return {
        "title": title,
        "authors": author_display,
        "category": category,
        "description": str(meta.get("description", "")),
        "highlights": [highlight_text],
        "persona_summary": persona_summary,
        "meta": {
            "title": title,
            "authors": author_display,
            "category": category,
            "description": str(meta.get("description", "")),
        },
    }
