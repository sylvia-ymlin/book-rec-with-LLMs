"""
P2: Zero-shot intent probing for cold-start users.

Uses LLM to infer categories, emotions, and keywords from a user's first query.
When user has no history, this helps seed preferences for faster convergence.
"""

import json
import re
from typing import Optional

from src.utils import setup_logger

logger = setup_logger(__name__)

# Categories we support (match router/metadata)
KNOWN_CATEGORIES = [
    "Fiction", "History", "Philosophy", "Science", "Art",
    "Biography", "Mystery", "Romance", "Fantasy", "Science Fiction",
    "Literary", "General",
]

EMOTION_KEYWORDS = [
    "happy", "sad", "suspenseful", "angry", "surprising",
    "heartbreaking", "uplifting", "thought-provoking", "relaxing",
]


def probe_intent(query: str, llm=None) -> dict:
    """
    Infer user intent from a short query (zero-shot, no history).

    Returns:
        dict with keys: categories, emotions, keywords, summary
    """
    if not query or not query.strip():
        return {"categories": [], "emotions": [], "keywords": [], "summary": ""}

    if llm is None:
        try:
            from src.core.llm import get_llm_model
            import os
            provider = os.getenv("LLM_PROVIDER", "ollama")
            api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else None
            llm = get_llm_model(provider=provider, api_key=api_key)
        except Exception as e:
            logger.warning(f"Intent prober: LLM not available ({e}), using rule-based fallback")
            return _rule_based_intent(query)

    prompt = f"""Analyze this book preference query and return JSON only.

Query: "{query.strip()}"

Extract:
- categories: list of book categories from {KNOWN_CATEGORIES} that match (max 3)
- emotions: list of emotions/moods from {EMOTION_KEYWORDS} that match (max 2)
- keywords: 2-4 short searchable keywords (e.g. "WWII", "detective", "love story")
- summary: one short sentence summarizing what the user wants

Return only valid JSON, no markdown:
{{"categories": [...], "emotions": [...], "keywords": [...], "summary": "..."}}"""

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "categories": data.get("categories", [])[:3],
                "emotions": data.get("emotions", [])[:2],
                "keywords": data.get("keywords", [])[:4],
                "summary": data.get("summary", "")[:200],
            }
    except Exception as e:
        logger.warning(f"Intent prober LLM failed: {e}")

    return _rule_based_intent(query)


def _rule_based_intent(query: str) -> dict:
    """Fallback when LLM unavailable: simple keyword matching."""
    lower = query.lower().strip()
    categories = []
    emotions = []
    keywords = []

    cat_map = {
        "fiction": "Fiction", "history": "History", "philosophy": "Philosophy",
        "science": "Science", "art": "Art", "mystery": "Mystery", "romance": "Romance",
        "fantasy": "Fantasy", "sci-fi": "Science Fiction", "biography": "Biography",
    }
    for k, v in cat_map.items():
        if re.search(r"\b" + re.escape(k) + r"\b", lower):
            categories.append(v)

    for e in EMOTION_KEYWORDS:
        if re.search(r"\b" + re.escape(e) + r"\b", lower):
            emotions.append(e)

    # Extract likely keywords (words 4+ chars, not common)
    stop = {"book", "books", "want", "like", "looking", "something", "that", "with", "the", "and"}
    words = [w for w in re.findall(r"\b\w{4,}\b", lower) if w not in stop][:4]
    keywords.extend(words)

    return {
        "categories": categories[:3] or ["General"],
        "emotions": emotions[:2],
        "keywords": keywords[:4],
        "summary": query[:150] if query else "",
    }
