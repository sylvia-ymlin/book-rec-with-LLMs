#!/usr/bin/env python3
"""
Entry script: Train YoutubeDNN Two-Tower recall model.

All training logic lives in YoutubeDNNRecall.fit(). This script loads data
and calls fit().

Usage:
    python scripts/model/train_youtube_dnn.py

Input:  data/rec/train.csv
Output: data/model/recall/youtube_dnn.pt, youtube_dnn_meta.pkl
        data/rec/item_map.pkl, user_sequences.pkl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import logging

from src.recall.embedding import YoutubeDNNRecall

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "rec" / "train.csv"
BOOKS_PATH = PROJECT_ROOT / "data" / "books_processed.csv"


def main():
    logger.info("Loading training data from %s...", TRAIN_PATH)
    df = pd.read_csv(TRAIN_PATH)
    logger.info("Loaded %d records.", len(df))

    model = YoutubeDNNRecall()
    model.fit(df, books_path=BOOKS_PATH)
    logger.info("YoutubeDNN training complete.")


if __name__ == "__main__":
    main()
