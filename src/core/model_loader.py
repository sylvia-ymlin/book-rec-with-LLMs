"""
Remote Model Loader

Downloads model artifacts from Hugging Face Hub at runtime.
This allows hosting large models separately from the Space repo,
bypassing the 1GB storage limit.

Usage:
    from src.core.model_loader import ensure_models_exist
    ensure_models_exist()  # Call once at startup
"""

import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# Configuration
HF_REPO_ID = os.getenv("HF_MODEL_REPO", "ymlin105/book-rec-models")
LOCAL_MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "model"

# Files to download (relative paths within the HF repo)
MODEL_FILES = [
    # Recall models
    "recall/itemcf.pkl",
    "recall/usercf.pkl",
    "recall/swing.pkl",
    "recall/item2vec.pkl",
    "recall/popularity.pkl",
    "recall/youtube_dnn.pt",
    "recall/youtube_dnn_meta.pkl",
    # SASRec
    "rec/sasrec_model.pth",
    # Ranking models
    "ranking/lgbm_ranker.txt",
    "ranking/xgb_ranker.json",
    "ranking/stacking_meta.pkl",
]


def ensure_models_exist():
    """
    Check if critical data artifacts exist locally; if not, download from HF Hub.
    Using snapshot_download for bulk restoration of indices and CSVs.
    """
    # Check for a few critical marker files to decide if we need a full sync
    critical_files = [
        "books.db",
        "recall_models.db",
        "chroma_db/chroma.sqlite3"
    ]
    
    missing = [f for f in critical_files if not (DATA_DIR / f).exists()]
    
    if not missing:
        logger.info("Critical data artifacts found locally, skipping bulk download.")
        return
    
    logger.info(f"Missing {missing}, performing bulk data restoration from HF Hub...")
    
    try:
        # Download the entire folder (respecting .gitignore is handled by the Hub)
        # Note: we use allow_patterns if we want to be specific, but for now 
        # we've curated the repo to only have what we need.
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=DATA_DIR,
            local_dir_use_symlinks=False,
            # Ignore raw data if it somehow got in
            ignore_patterns=["raw/*", "sft/*", "*.bak"],
        )
        logger.info("Data restoration complete.")
    except Exception as e:
        logger.error(f"Failed to restore data from HF Hub: {e}")
        # Don't raise here, allow app to try starting anyway (Survival Mode will kick in)


def download_all_models():
    """
    Force download all models (useful for rebuilding cache).
    """
    logger.info(f"Downloading all models from {HF_REPO_ID}...")
    
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
    )
    
    logger.info("All models downloaded.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_models_exist()
