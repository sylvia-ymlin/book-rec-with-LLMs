#!/usr/bin/env python3
"""
Build User Sequences for Sequential Models (SASRec, YoutubeDNN)

Converts user interaction history into padded sequences for training.

TIME-SPLIT (strict): Uses train.csv ONLY for sequences and item_map.
This prevents leakage when Ranking uses SASRec embeddings as features:
- Val/test samples must not appear in user history when computing sasrec_score.
- Recall models (SASRec, YoutubeDNN) overwrite these with their own train-only output.

Usage:
    python scripts/data/build_sequences.py

Input:
    - data/rec/train.csv (val.csv, test.csv exist but are NOT used for sequences)

Output:
    - data/rec/user_sequences.pkl  (Dict[user_id, List[item_id]]) — train-only
    - data/rec/item_map.pkl        (Dict[isbn, item_id]) — train-only
"""

import pandas as pd
import pickle
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_sequences(data_dir="data/rec", max_len=50):
    """
    Build user sequences from train.csv only (strict time-split).
    Val/test are excluded to avoid leakage in ranking features (sasrec_score).
    """
    logger.info("Building user sequences (train-only, time-split)...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")

    # 1. Item map from train only (matches SASRec/YoutubeDNN)
    items = train_df["isbn"].unique()
    item_map = {isbn: i + 1 for i, isbn in enumerate(items)}
    logger.info("  Items (train): %d", len(item_map))

    # 2. User history from train only (no val/test)
    user_history = {}
    if "timestamp" in train_df.columns:
        train_df = train_df.sort_values(["user_id", "timestamp"])
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="  Processing"):
        u = str(row["user_id"])
        item = item_map.get(row["isbn"])
        if item is None:
            continue
        if u not in user_history:
            user_history[u] = []
        user_history[u].append(item)

    final_seqs = {u: hist[-max_len:] for u, hist in user_history.items()}
    logger.info("  Users: %d", len(final_seqs))

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "item_map.pkl", "wb") as f:
        pickle.dump(item_map, f)
    with open(data_dir / "user_sequences.pkl", "wb") as f:
        pickle.dump(final_seqs, f)
    logger.info("Sequence data saved (train-only).")
    
if __name__ == "__main__":
    build_sequences()
