import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data Paths
# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
# User-generated data (Persistent)
# On HF Spaces with persistent storage, set USER_DATA_PATH=/data
_user_data_path = os.getenv("USER_DATA_PATH")
USER_DATA_DIR = Path(_user_data_path) if _user_data_path else DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR # Alias for clearer intent
BOOKS_CSV = DATA_DIR / "books_with_emotions.csv"
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
