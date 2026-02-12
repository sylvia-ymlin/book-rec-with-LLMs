#!/usr/bin/env python3
"""
Entry script: Train SASRec sequential recommendation model.

All training logic lives in SASRecRecall.fit(). This script loads data
and calls fit().

Usage:
    python scripts/model/train_sasrec.py

Input:  data/rec/train.csv
Output: data/model/rec/sasrec_model.pth
        data/rec/user_seq_emb.pkl, item_map.pkl, user_sequences.pkl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import logging

from src.recsys.recall.sasrec_recall import SASRecRecall

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "rec" / "train.csv"


def main():
    logger.info("Loading training data from %s...", TRAIN_PATH)
    df = pd.read_csv(TRAIN_PATH)
    logger.info("Loaded %d records.", len(df))

    model = SASRecRecall()
    model.fit(df)
    logger.info("SASRec training complete.")


if __name__ == "__main__":
    main()
