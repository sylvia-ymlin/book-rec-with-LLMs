import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CONFIG_DIR = PROJECT_ROOT / "config"

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
# On HF Spaces with persistent storage, set USER_DATA_PATH=/data
_user_data_path = os.getenv("USER_DATA_PATH")
USER_DATA_DIR = Path(_user_data_path) if _user_data_path else DATA_DIR
REVIEW_HIGHLIGHTS_TXT = DATA_DIR / "review_highlights.txt"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Assets
ASSETS_DIR = PROJECT_ROOT / "assets"
COVER_NOT_FOUND = ASSETS_DIR / "cover-not-found.jpg"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = 3600  # 1 hour

# App Settings
TOP_K_INITIAL = 50
TOP_K_FINAL = 10

# Latency: Rerank candidate cap (lower = faster, LATENCY_OPTIMIZATION.md)
RERANK_CANDIDATES_MAX = int(os.getenv("RERANK_CANDIDATES_MAX", "20"))

# Reranker backend: cross_encoder | onnx | colbert (onnx ~2x faster, colbert optional)
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "onnx")

# Debug mode: set DEBUG=1 to enable verbose logging (research prototype style)
DEBUG = os.getenv("DEBUG", "0") == "1"


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
        except Exception:
            pass
    return defaults


_ROUTER_CFG = _load_router_config()

# Dependencies can override via ROUTER_CONFIG_PATH for alternate config
_path_override = os.getenv("ROUTER_CONFIG_PATH")
if _path_override and Path(_path_override).exists():
    try:
        _ROUTER_CFG = {**_ROUTER_CFG, **json.loads(Path(_path_override).read_text(encoding="utf-8"))}
    except Exception:
        pass

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
