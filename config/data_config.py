"""
Data Pipeline Configuration

Centralized path and parameter management for data processing scripts.
All scripts should import from here to ensure consistency.
"""

from pathlib import Path
import os

# =============================================================================
# Directory Structure
# =============================================================================

# Find project root by looking for Makefile or .git
def _find_project_root():
    """Find project root directory."""
    # Try from config file location first
    config_path = Path(__file__).resolve().parent.parent
    if (config_path / "Makefile").exists():
        return config_path
    
    # Try from current working directory
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "Makefile").exists() or (parent / ".git").exists():
            return parent
    
    # Fallback
    return cwd

PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"


# Raw data (input)
RAW_DIR = DATA_DIR / "raw"
RAW_BOOKS = RAW_DIR / "books_data.csv"
RAW_RATINGS = RAW_DIR / "Books_rating.csv"

# Processed data (intermediate)
BOOKS_BASIC_INFO = DATA_DIR / "books_basic_info.csv"
BOOKS_PROCESSED = DATA_DIR / "books_processed.csv"
REVIEW_HIGHLIGHTS = DATA_DIR / "review_highlights.txt"
REVIEW_CHUNKS = DATA_DIR / "review_chunks.jsonl"

# RecSys data
REC_DIR = DATA_DIR / "rec"
TRAIN_CSV = REC_DIR / "train.csv"
VAL_CSV = REC_DIR / "val.csv"
TEST_CSV = REC_DIR / "test.csv"
USER_SEQUENCES = REC_DIR / "user_sequences.pkl"
ITEM_MAP = REC_DIR / "item_map.pkl"
ACTIVE_USERS = REC_DIR / "active_users.csv"

# Vector indices
CHROMA_DB = DATA_DIR / "chroma_db"
CHROMA_CHUNKS = DATA_DIR / "chroma_chunks"

# Models
MODEL_DIR = DATA_DIR / "model"
RECALL_DIR = MODEL_DIR / "recall"
RANKING_DIR = MODEL_DIR / "ranking"

ITEMCF_MODEL = RECALL_DIR / "itemcf.pkl"
USERCF_MODEL = RECALL_DIR / "usercf.pkl"
YOUTUBE_DNN_MODEL = RECALL_DIR / "youtube_dnn.pt"
YOUTUBE_DNN_META = RECALL_DIR / "youtube_dnn_meta.pkl"
SASREC_MODEL = RECALL_DIR / "sasrec.pt"
ITEM2VEC_MODEL = RECALL_DIR / "item2vec.pkl"
LGBM_RANKER = RANKING_DIR / "lgbm_ranker.txt"
XGB_RANKER = RANKING_DIR / "xgb_ranker.json"
STACKING_META = RANKING_DIR / "stacking_meta.pkl"

# User data
USER_PROFILES = DATA_DIR / "user_profiles.json"

# =============================================================================
# Processing Parameters
# =============================================================================

# Data split
MIN_USER_INTERACTIONS = 3  # Minimum ratings per user

# Sequence building
MAX_SEQUENCE_LENGTH = 50

# Emotion generation
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_LABELS = ["joy", "sadness", "fear", "anger", "surprise"]

# Tag generation
TAG_TOP_N = 8
TAG_MAX_FEATURES = 60000
TAG_MIN_DF = 5
TAG_MAX_DF = 0.5

# Chunking
CHUNK_MIN_LEN = 50
CHUNK_MAX_LEN = 300

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Training
YOUTUBE_DNN_EPOCHS = 50
YOUTUBE_DNN_BATCH = 512
YOUTUBE_DNN_EMBED_DIM = 64

SASREC_EPOCHS = 30
SASREC_BATCH = 256
SASREC_EMBED_DIM = 64

# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dirs():
    """Create all required directories."""
    for d in [RAW_DIR, REC_DIR, CHROMA_DB, CHROMA_CHUNKS, RECALL_DIR, RANKING_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_device():
    """Get the best available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
