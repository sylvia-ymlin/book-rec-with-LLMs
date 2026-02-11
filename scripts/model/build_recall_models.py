#!/usr/bin/env python3
"""
Entry script: Build recall models (ItemCF, UserCF, Swing, Popularity, Item2Vec).

All training logic lives in src/recall/*.fit(). This script only loads data,
imports models, and calls fit().

Usage:
    python scripts/model/build_recall_models.py

Input:  data/rec/train.csv (columns: user_id, isbn, rating, timestamp)
Output: data/model/recall/*.pkl, data/recall_models.db (ItemCF)
"""

import sys
from pathlib import Path

# Run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import logging

from src.recall.itemcf import ItemCF
from src.recall.usercf import UserCF
from src.recall.swing import Swing
from src.recall.popularity import PopularityRecall
from src.recall.item2vec import Item2Vec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "rec" / "train.csv"


def main():
    logger.info("Loading training data from %s...", TRAIN_PATH)
    df = pd.read_csv(TRAIN_PATH)
    logger.info("Loaded %d records.", len(df))

    logger.info("--- Training ItemCF ---")
    ItemCF().fit(df)

    logger.info("--- Training UserCF ---")
    UserCF().fit(df)

    logger.info("--- Training Swing ---")
    Swing().fit(df)

    logger.info("--- Training Popularity ---")
    PopularityRecall().fit(df)

    logger.info("--- Training Item2Vec ---")
    Item2Vec().fit(df)

    logger.info("Recall models built and saved successfully.")


if __name__ == "__main__":
    main()
