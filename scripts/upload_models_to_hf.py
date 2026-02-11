#!/usr/bin/env python3
"""
Upload models to Hugging Face Hub.

This script uploads all model files to a HF Dataset repository,
which can then be downloaded at runtime by the Space.

Usage:
    # First, login to HF
    huggingface-cli login
    
    # Then run this script
    python scripts/upload_models_to_hf.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = os.getenv("HF_MODEL_REPO", "ymlin105/book-rec-models")
LOCAL_MODEL_DIR = Path(__file__).parent.parent / "data" / "model"

# Files to upload
MODEL_FILES = [
    "recall/itemcf.pkl",
    "recall/usercf.pkl", 
    "recall/swing.pkl",
    "recall/item2vec.pkl",
    "recall/popularity.pkl",
    "recall/youtube_dnn.pt",
    "recall/youtube_dnn_meta.pkl",
    "rec/sasrec_model.pth",
    "ranking/lgbm_ranker.txt",
    "ranking/xgb_ranker.json",
    "ranking/stacking_meta.pkl",
]


def main():
    api = HfApi()
    
    # Create repo if it doesn't exist
    print(f"Creating/checking repo: {REPO_ID}")
    try:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")
    
    # Upload data folder
    # Excluding raw data and user profiles to keep it clean and under limit
    print(f"Uploading data folder to {REPO_ID}...")
    
    # Define directory to upload (parent of model, but includes other essentials)
    DATA_PATH = Path(__file__).parent.parent / "data"
    
    ignore_patterns = [
        "raw/*",
        "rec/*.csv",  # Exclude training data
        "user_profiles.json",
        "users.json",
        "*.bak",
        "*.log",
        "sft/*",
        "chroma_chunks/*" # These are just intermediate
    ]
    
    api.upload_folder(
        folder_path=str(DATA_PATH),
        repo_id=REPO_ID,
        repo_type="dataset",
        ignore_patterns=ignore_patterns,
        delete_patterns=None, # Don't delete existing files in repo for safety
    )
    
    print("\n🎉 All data and models uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
