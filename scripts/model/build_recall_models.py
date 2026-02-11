#!/usr/bin/env python3
"""
Build Traditional Recall Models (ItemCF, UserCF, Swing, Popularity)

Trains collaborative filtering and popularity-based recall models.
These are CPU-friendly and provide strong baselines.

Usage:
    python scripts/model/build_recall_models.py

Input:
    - data/rec/train.csv

Output:
    - data/model/recall/itemcf.pkl   (~1.4 GB)
    - data/model/recall/usercf.pkl   (~70 MB)
    - data/model/recall/swing.pkl
    - data/model/recall/popularity.pkl

Algorithms:
    - ItemCF: Co-rating similarity with direction weight (forward=1.0, backward=0.7)
    - UserCF: User similarity (Jaccard + activity penalty)
    - Swing: User-pair overlap weighting for substitute relationships
    - Popularity: Rating count with time decay
"""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import logging
from src.recall.itemcf import ItemCF
from src.recall.usercf import UserCF
from src.recall.swing import Swing
from src.recall.popularity import PopularityRecall

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading training data...")
    df = pd.read_csv('data/rec/train.csv')
    
    # 1. ItemCF
    logger.info("--- Training ItemCF ---")
    itemcf = ItemCF()
    if itemcf.load():
        logger.info("ItemCF model already exists, skipping training.")
    else:
        itemcf.fit(df)
    
    # 2. UserCF
    logger.info("--- Training UserCF ---")
    # For UserCF, using full data might be slow if many users/items.
    # The current implementation has hot-item pruning (limit=2000).
    # 1M records, 114k users. 
    usercf = UserCF()
    if usercf.load():
         # Force retrain if we optimized logic? No, load() returns True if exists.
         # But I just changed logic, so I want to RETRAIN UserCF.
         pass
    
    # Just force retrain UserCF for now since I optimized it
    usercf.fit(df)
    
    # 3. Swing
    logger.info("--- Training Swing ---")
    swing = Swing()
    swing.fit(df)

    # 4. Popularity
    logger.info("--- Training Popularity ---")
    pop = PopularityRecall()
    pop.fit(df)
    
    logger.info("Recall models built and saved successfully!")

if __name__ == "__main__":
    main()
