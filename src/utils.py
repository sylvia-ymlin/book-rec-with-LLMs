import logging
import sys
import re
import html

def setup_logger(name: str):
    """Configure and return a logger. Use DEBUG=1 for verbose output."""
    from src.config import DEBUG
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def summarize_description(text: str, max_sentences: int = 2, max_chars: int = 240) -> str:
    """Create a clean, sentence-based summary for a book description.

    - Decodes HTML entities (e.g., &amp; → &)
    - Normalizes whitespace
    - Truncates by complete sentences (not raw words)
    - Applies a soft character cap with an ellipsis if needed
    """
    if not text:
        return "—"

    # Decode HTML entities and normalize whitespace
    cleaned = html.unescape(str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return "—"

    # Split into sentences on punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    selected: list[str] = []
    total_len = 0
    for s in sentences:
        if not s:
            continue
        # Tentatively add sentence if within limits
        if len(selected) < max_sentences and (total_len + len(s) + (1 if selected else 0)) <= max_chars:
            selected.append(s)
            total_len += len(s) + (1 if selected else 0)
        else:
            break

    summary = " ".join(selected).strip()
    if not summary:
        # Fallback: hard trim characters with ellipsis
        summary = cleaned[: max_chars].rstrip()
        if len(cleaned) > max_chars:
            summary = summary.rsplit(" ", 1)[0].rstrip() + "…"
        return summary

    # Ensure soft char cap
    if len(summary) > max_chars:
        summary = summary[: max_chars].rstrip()
        summary = summary.rsplit(" ", 1)[0].rstrip() + "…"

    return summary


def enrich_book_metadata(meta: dict, isbn: str) -> dict:
    """
    Enrich book metadata with dynamic cover fetching if missing.
    Mutates and returns the meta dictionary.
    """
    if not meta:
        meta = {}
    
    # 1. Get available metadata
    title = meta.get("title")
    thumbnail = meta.get("thumbnail")
    author = meta.get("authors", "Unknown")
    
    # 2. Validation Check
    is_valid_thumb = thumbnail and str(thumbnail).lower() not in ["nan", "none", "", "null"] and "/assets/cover-not-found.jpg" not in str(thumbnail) and "cover-not-found" not in str(thumbnail)
    
    # 3. Fetch if needed
    if not title or not is_valid_thumb:
        # Lazy import to avoid circular dependency
        from src.cover_fetcher import fetch_book_cover
        
        fetched_cover, fetched_authors, fetched_desc = fetch_book_cover(str(isbn))
        
        # Update if we found better data
        if not is_valid_thumb and "cover-not-found" not in fetched_cover:
            meta["thumbnail"] = fetched_cover
        
        if not title:
             meta["title"] = f"Book {isbn}"
        
        if author == "Unknown" and fetched_authors != "Unknown":
            meta["authors"] = fetched_authors
            
    # 4. Final Fallback
    final_thumb = meta.get("thumbnail")
    if not final_thumb or str(final_thumb).lower() in ["nan", "none", "", "null"] or "cover-not-found" in str(final_thumb):
         meta["thumbnail"] = "/content/cover-not-found.jpg"
         
    return meta
