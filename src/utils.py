import logging
import sys
import re
import html

def setup_logger(name: str):
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
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
