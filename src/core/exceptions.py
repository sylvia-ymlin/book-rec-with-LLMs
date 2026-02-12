"""
Exception hierarchy for recommendation system.

STRATEGY:
  - Retrieval/Recall failures: log, return empty list (graceful degradation).
  - Metadata/DB failures: log, propagate if critical; return empty/None for optional paths.
  - External API (web search, LLM): catch specific exceptions (Timeout, ConnectionError),
    log, fallback to cached or empty; avoid bare `except Exception` swallowing bugs.
  - Never use bare `except:` — always catch specific types or `Exception` with explicit log.
"""


class RecommendationError(Exception):
    """Base for all recommendation-related errors."""


class VectorDBError(RecommendationError):
    """ChromaDB, FTS5, or vector search failure."""


class MetadataStoreError(RecommendationError):
    """SQLite/metadata lookup failure."""


class RecallError(RecommendationError):
    """Recall model load or inference failure."""


class ExternalAPIError(RecommendationError):
    """Web search, Google Books API, or other external service failure."""
