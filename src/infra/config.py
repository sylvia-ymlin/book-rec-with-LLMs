"""
Unified configuration. Uses pydantic-settings for env loading and validation.
"""
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent.absolute()


class Settings(BaseSettings):
    """Single source of truth for app config. Env vars override defaults."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_root: Path = Field(default_factory=_project_root)
    user_data_path: Optional[str] = Field(default=None, validation_alias="USER_DATA_PATH")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token: Optional[str] = Field(default=None, validation_alias="HUGGINGFACEHUB_API_TOKEN")
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    cache_ttl: int = 3600
    top_k_initial: int = 50
    top_k_final: int = 10
    rerank_candidates_max: int = Field(default=20, validation_alias="RERANK_CANDIDATES_MAX")
    reranker_backend: str = Field(default="onnx", validation_alias="RERANKER_BACKEND")
    debug: bool = Field(default=False, validation_alias="DEBUG")
    # A/B testing: enable experiment assignment for diversity_rerank etc.
    ab_experiments_enabled: bool = Field(default=False, validation_alias="AB_EXPERIMENTS_ENABLED")
    # RAG path: apply diversity reranker (MMR + category constraint) to reduce homogeneity.
    enable_rag_diversity: bool = Field(default=True, validation_alias="ENABLE_RAG_DIVERSITY")


# -----------------------------------------------------------------------------
# Algorithm constants (no magic numbers in core logic)
# -----------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion): score = 1 / (k + rank). Larger k = less rank sensitivity.
RRF_K: int = 60
# Diversity reranker: MMR lambda (relevance weight), popularity penalty, category cap.
MMR_LAMBDA_DEFAULT: float = 0.75
POPULARITY_GAMMA_DEFAULT: float = 0.1
MAX_PER_CATEGORY_DEFAULT: int = 3
# Router: trigger web fallback when local results < threshold.
FRESHNESS_THRESHOLD: int = 3
# Small-to-Big: over-retrieve chunks by factor before mapping to parent books.
SMALL_TO_BIG_OVER_RETRIEVE_FACTOR: int = 3
# Hybrid search: alpha=0 -> dense-only, alpha=1 -> sparse-only.
HYBRID_DEFAULT_ALPHA: float = 0.5


_settings = Settings()

# Re-export for backward compatibility (existing code imports these directly)
PROJECT_ROOT = _settings.project_root
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
USER_DATA_DIR = Path(_settings.user_data_path) if _settings.user_data_path else DATA_DIR
REVIEW_HIGHLIGHTS_TXT = DATA_DIR / "review_highlights.txt"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
ASSETS_DIR = PROJECT_ROOT / "assets"
COVER_NOT_FOUND = ASSETS_DIR / "cover-not-found.jpg"
EMBEDDING_MODEL = _settings.embedding_model
HF_TOKEN = _settings.hf_token
REDIS_URL = _settings.redis_url
CACHE_TTL = _settings.cache_ttl
TOP_K_INITIAL = _settings.top_k_initial
TOP_K_FINAL = _settings.top_k_final
RERANK_CANDIDATES_MAX = _settings.rerank_candidates_max
RERANKER_BACKEND = _settings.reranker_backend
DEBUG = _settings.debug
AB_EXPERIMENTS_ENABLED = _settings.ab_experiments_enabled
ENABLE_RAG_DIVERSITY = _settings.enable_rag_diversity
# Algorithm constants (defined above, re-exported for imports)
# RRF_K, MMR_LAMBDA_DEFAULT, POPULARITY_GAMMA_DEFAULT, MAX_PER_CATEGORY_DEFAULT,
# FRESHNESS_THRESHOLD, SMALL_TO_BIG_OVER_RETRIEVE_FACTOR, HYBRID_DEFAULT_ALPHA


def get_settings() -> Settings:
    """Return the unified Settings instance (for new code preferring typed access)."""
    return _settings


def _load_router_config() -> dict:
    """Load router keywords from config/router.json. Env overrides for ops flexibility."""
    defaults = {
        "detail_keywords": [
            "twist", "ending", "spoiler", "readers", "felt", "cried", "hated", "loved",
            "review", "opinion", "think", "unreliable", "narrator", "realize", "find out",
        ],
        "freshness_keywords": [
            "new", "newest", "latest", "recent", "modern", "contemporary", "current",
        ],
        "strong_freshness_keywords": ["newest", "latest"],
        "natural_language_keywords": [
            "like", "similar", "recommend", "want", "looking", "books", "something",
            "suggest", "recommendations", "after", "read", "if", "liked",
        ],
    }
    path = CONFIG_DIR / "router.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {**defaults, **data}
        except (json.JSONDecodeError, OSError) as e:
            # Fallback to defaults if config invalid or unreadable
            import logging
            logging.getLogger(__name__).warning("router.json load failed: %s - using defaults", e)
    return defaults


_ROUTER_CFG = _load_router_config()

# Dependencies can override via ROUTER_CONFIG_PATH for alternate config
_path_override = os.getenv("ROUTER_CONFIG_PATH")
if _path_override and Path(_path_override).exists():
    try:
        _ROUTER_CFG = {**_ROUTER_CFG, **json.loads(Path(_path_override).read_text(encoding="utf-8"))}
    except (json.JSONDecodeError, OSError) as e:
        import logging
        logging.getLogger(__name__).warning("ROUTER_CONFIG_PATH load failed: %s", e)

# Env: ROUTER_DETAIL_KEYWORDS = "twist,ending,spoiler,..." (comma-separated) overrides config
_DETAIL_KW_RAW = os.getenv("ROUTER_DETAIL_KEYWORDS", "")
ROUTER_DETAIL_KEYWORDS: frozenset[str] = (
    frozenset(w.strip().lower() for w in _DETAIL_KW_RAW.split(",") if w.strip())
    if _DETAIL_KW_RAW
    else frozenset(str(k).lower() for k in _ROUTER_CFG.get("detail_keywords", []))
)

ROUTER_FRESHNESS_KEYWORDS: frozenset[str] = frozenset(
    str(k).lower() for k in _ROUTER_CFG.get("freshness_keywords", [])
)
ROUTER_STRONG_FRESHNESS_KEYWORDS: frozenset[str] = frozenset(
    str(k).lower() for k in _ROUTER_CFG.get("strong_freshness_keywords", [])
)
ROUTER_NL_KEYWORDS: frozenset[str] = frozenset(
    str(k).lower() for k in _ROUTER_CFG.get("natural_language_keywords", [])
)

