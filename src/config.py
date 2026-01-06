import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
BOOKS_CSV = DATA_DIR / "books_with_emotions.csv"
DESCRIPTIONS_TXT = DATA_DIR / "books_descriptions.txt"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Assets
ASSETS_DIR = PROJECT_ROOT / "assets"
COVER_NOT_FOUND = ASSETS_DIR / "cover-not-found.jpg"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# App Settings
TOP_K_INITIAL = 50
TOP_K_FINAL = 10
