#!/usr/bin/env python3
"""
Build Traditional Recall Models (ItemCF, UserCF, Swing, Popularity, Item2Vec)

Trains collaborative filtering, embedding-based, and popularity recall models.
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
    - data/model/recall/item2vec.pkl

Algorithms:
    - ItemCF: Co-rating similarity with direction weight (forward=1.0, backward=0.7)
    - UserCF: User similarity (Jaccard + activity penalty)
    - Swing: User-pair overlap weighting for substitute relationships
    - Popularity: Rating count with time decay
    - Item2Vec: Word2Vec (Skip-gram) on user interaction sequences
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
from src.recall.item2vec import Item2Vec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading training data...")
    df = pd.read_csv('data/rec/train.csv')
    
    # 1. ItemCF (force retrain — direction weight updated)
    logger.info("--- Training ItemCF ---")
    itemcf = ItemCF()
    itemcf.fit(df)
    
    # 2. UserCF
    logger.info("--- Training UserCF ---")
    usercf = UserCF()
    usercf.fit(df)
    
    # 3. Swing
    logger.info("--- Training Swing ---")
    swing = Swing()
    swing.fit(df)

    # 4. Popularity
    logger.info("--- Training Popularity ---")
    pop = PopularityRecall()
    pop.fit(df)
    
    # 5. Item2Vec
    logger.info("--- Training Item2Vec ---")
    item2vec = Item2Vec()
    item2vec.fit(df)

    logger.info("Recall models built and saved successfully!")

if __name__ == "__main__":
    main()
