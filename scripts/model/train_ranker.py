#!/usr/bin/env python3
"""
Train LightGBM LambdaRank Model for Personalized Recommendations

Learning-to-Rank model that optimizes NDCG directly.
Combines features from ItemCF, UserCF, SASRec, Swing, and user/item statistics.

Usage:
    python scripts/model/train_ranker.py

Input:
    - data/rec/val.csv                (positive samples)
    - data/rec/train.csv              (for negative sampling)
    - data/model/recall/*.pkl         (recall model features)

Output:
    - data/model/ranking/lgbm_ranker.txt

Features:
    - User stats: count, mean rating, std
    - Item stats: count, mean rating, std
    - Content: description length diff, author affinity
    - SASRec: embedding similarity
    - Last-N: max/min/mean similarity to recent items
    - Category: affinity indicator
    - ItemCF/UserCF interaction scores

Training:
    - Positive: user-item pairs from val.csv (label=1)
    - Negative: random sampling (4x negatives per positive, label=0)
    - Grouped by user for LambdaRank
    - Objective: lambdarank, metric: ndcg
"""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from pathlib import Path
from src.ranking.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_ranker_data(data_dir='data/rec', neg_ratio=4):
    """
    Construct training data for ranker, grouped by user for LTR.
    Returns DataFrame sorted by user_id (required for group parameter).
    """
    logger.info("Building ranker training data...")
    val_df = pd.read_csv(f'{data_dir}/val.csv')

    all_items = pd.read_csv(f'{data_dir}/train.csv')['isbn'].unique()

    rows = []
    for _, row in val_df.iterrows():
        user_id = row['user_id']
        pos_isbn = row['isbn']

        # 1 positive
        rows.append({'user_id': user_id, 'isbn': pos_isbn, 'label': 1})

        # N negatives
        neg_items = np.random.choice(all_items, size=neg_ratio, replace=False)
        for neg_isbn in neg_items:
            rows.append({'user_id': user_id, 'isbn': neg_isbn, 'label': 0})

    train_data = pd.DataFrame(rows)
    # Sort by user_id so group parameter aligns
    train_data = train_data.sort_values('user_id').reset_index(drop=True)

    # Build group array: each user has (1 + neg_ratio) candidates
    group_size = 1 + neg_ratio
    n_groups = len(train_data) // group_size
    group = [group_size] * n_groups

    return train_data, group


def train_ranker():
    data_dir = Path('data/rec')
    model_dir = Path('data/model/ranking')
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    train_samples, group = build_ranker_data(str(data_dir))
    logger.info(f"Training samples: {len(train_samples)}, groups: {len(group)}")

    # 2. Generate Features
    fe = FeatureEngineer(data_dir=str(data_dir), model_dir='data/model/recall')

    logger.info("Generating features...")
    X_y = fe.create_dateset(train_samples)

    # 3. Train LGBMRanker
    features = [c for c in X_y.columns if c not in ['label', 'user_id', 'isbn']]
    X = X_y[features]
    y = X_y['label']

    logger.info(f"Training LGBMRanker with {len(features)} features: {features}")

    model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=20,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y, group=group)

    # 4. Save
    model_path = model_dir / 'lgbm_ranker.txt'
    model.booster_.save_model(str(model_path))
    logger.info(f"Ranker saved to {model_path}")

    # Feature Importance
    importance = model.feature_importances_
    for i, score in enumerate(importance):
        logger.info(f"Feature {features[i]}: {score}")

if __name__ == "__main__":
    train_ranker()
