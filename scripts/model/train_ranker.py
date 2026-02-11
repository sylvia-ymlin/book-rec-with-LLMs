#!/usr/bin/env python3
"""
Train Ranking Models for Personalized Recommendations

Supports two modes:
    1. Standard: LGBMRanker (LambdaRank) single model
    2. Stacking: LGBMRanker + XGBClassifier -> LogisticRegression meta-learner

Usage:
    python scripts/model/train_ranker.py              # Standard mode
    python scripts/model/train_ranker.py --stacking    # Stacking mode

Input:
    - data/rec/val.csv                (positive samples)
    - data/rec/train.csv              (for fallback random negatives)
    - data/model/recall/*.pkl         (recall models for hard negative mining)

Output (Standard):
    - data/model/ranking/lgbm_ranker.txt

Output (Stacking):
    - data/model/ranking/lgbm_ranker.txt   (full retrained LGB)
    - data/model/ranking/xgb_ranker.json   (full retrained XGB)
    - data/model/ranking/stacking_meta.pkl (LogisticRegression meta-model)

Negative Sampling Strategy:
    - Hard negatives: items from recall results that are NOT the positive
    - Random negatives: fill remaining slots if recall returns too few
    - This teaches the ranker to distinguish between "close but wrong" vs "right"
"""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import xgboost as xgb
import logging
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from src.ranking.features import FeatureEngineer
from src.recall.fusion import RecallFusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_ranker_data(data_dir='data/rec', model_dir='data/model/recall', neg_ratio=4, max_samples=20000):
    """
    Construct training data with hard negative sampling.

    For each user in val.csv (sampled to max_samples for speed):
        - Positive: the actual item from val.csv (label=1)
        - Hard negatives: top items recalled by the system but NOT the positive
        - Random negatives: fill if recall gives fewer than neg_ratio candidates

    Returns:
        train_data: DataFrame [user_id, isbn, label]
        group: list of group sizes for LambdaRank
    """
    logger.info("Building ranker training data with hard negatives...")
    val_df = pd.read_csv(f'{data_dir}/val.csv')
    all_items = pd.read_csv(f'{data_dir}/train.csv')['isbn'].unique()

    # Sample for speed — 20K users is sufficient for LTR training
    if len(val_df) > max_samples:
        logger.info(f"Sampling {max_samples} from {len(val_df)} val rows for speed")
        val_df = val_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    # Load recall models for hard negative mining
    logger.info("Loading recall models for hard negative mining...")
    fusion = RecallFusion(data_dir, model_dir)
    fusion.load_models()

    rows = []
    group = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Mining hard negatives"):
        user_id = row['user_id']
        pos_isbn = row['isbn']

        # 1. Positive
        user_rows = [{'user_id': user_id, 'isbn': pos_isbn, 'label': 1}]

        # 2. Hard negatives from recall
        try:
            recall_items = fusion.get_recall_items(user_id, k=50)
            hard_negs = [item for item, _ in recall_items if item != pos_isbn]
            hard_negs = hard_negs[:neg_ratio]
        except Exception:
            hard_negs = []

        for neg_isbn in hard_negs:
            user_rows.append({'user_id': user_id, 'isbn': neg_isbn, 'label': 0})

        # 3. Fill with random negatives if not enough
        n_remaining = neg_ratio - len(hard_negs)
        if n_remaining > 0:
            random_negs = np.random.choice(all_items, size=n_remaining, replace=False)
            for neg_isbn in random_negs:
                user_rows.append({'user_id': user_id, 'isbn': neg_isbn, 'label': 0})

        rows.extend(user_rows)
        group.append(len(user_rows))

    train_data = pd.DataFrame(rows)
    logger.info(f"Built {len(train_data)} samples, {len(group)} groups")
    return train_data, group


def train_ranker():
    data_dir = Path('data/rec')
    model_dir = Path('data/model/ranking')
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    train_samples, group = build_ranker_data(
        str(data_dir), model_dir='data/model/recall', neg_ratio=4
    )
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


def train_stacking():
    """
    Train Level-1 models (LGBMRanker + XGBClassifier) via GroupKFold CV
    to produce out-of-fold (OOF) predictions, then train Level-2 meta-learner
    (LogisticRegression) to combine them.

    Architecture:
        Level-1: LGBMRanker (lambdarank scores) + XGBClassifier (probabilities)
        Level-2: LogisticRegression([lgb_score, xgb_score]) -> final probability
    """
    data_dir = Path('data/rec')
    model_dir = Path('data/model/ranking')
    model_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Prepare Data (reuse existing build_ranker_data)
    # =========================================================================
    train_samples, group = build_ranker_data(
        str(data_dir), model_dir='data/model/recall', neg_ratio=4
    )
    logger.info(f"Stacking training samples: {len(train_samples)}, groups: {len(group)}")

    # Generate Features
    fe = FeatureEngineer(data_dir=str(data_dir), model_dir='data/model/recall')
    logger.info("Generating features for stacking...")
    X_y = fe.create_dateset(train_samples)

    features = [c for c in X_y.columns if c not in ['label', 'user_id', 'isbn']]
    X = X_y[features].values
    y = X_y['label'].values

    logger.info(f"Stacking features ({len(features)}): {features}")

    # =========================================================================
    # 2. Build group_ids array for GroupKFold
    # =========================================================================
    # group is [5, 5, 5, ...] — each entry = # samples per user query
    # GroupKFold needs a group_id per sample
    group_ids = np.repeat(np.arange(len(group)), group)
    group_array = np.array(group)

    # =========================================================================
    # 3. K-Fold Cross-Validation for OOF Predictions
    # =========================================================================
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))

    logger.info(f"Running {n_splits}-fold GroupKFold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=group_ids)):
        logger.info(f"--- Fold {fold + 1}/{n_splits} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Reconstruct group sizes for train fold
        # GroupKFold keeps entire groups together, count per group_id
        train_group_ids = group_ids[train_idx]
        train_group_counts = Counter(train_group_ids)
        seen = set()
        train_groups = []
        for gid in train_group_ids:
            if gid not in seen:
                seen.add(gid)
                train_groups.append(train_group_counts[gid])

        # --- Level-1 Model A: LGBMRanker ---
        lgb_model = lgb.LGBMRanker(
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
        lgb_model.fit(X_train, y_train, group=train_groups)
        oof_lgb[val_idx] = lgb_model.predict(X_val)

        # --- Level-1 Model B: XGBClassifier ---
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

        logger.info(f"  Fold {fold+1} OOF — LGB mean: {oof_lgb[val_idx].mean():.4f}, "
                     f"XGB mean: {oof_xgb[val_idx].mean():.4f}")

    # =========================================================================
    # 4. Train Level-2 Meta-Learner on OOF predictions
    # =========================================================================
    logger.info("Training Level-2 meta-learner (LogisticRegression)...")
    meta_features = np.column_stack([oof_lgb, oof_xgb])

    meta_model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
    )
    meta_model.fit(meta_features, y)

    logger.info(f"Meta-learner coefficients: LGB={meta_model.coef_[0][0]:.4f}, "
                f"XGB={meta_model.coef_[0][1]:.4f}, "
                f"intercept={meta_model.intercept_[0]:.4f}")

    # =========================================================================
    # 5. Retrain Level-1 models on FULL data (for inference)
    # =========================================================================
    logger.info("Retraining Level-1 models on full data...")

    # Full LGBMRanker
    full_lgb = lgb.LGBMRanker(
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
    full_lgb.fit(X, y, group=group)

    lgb_path = model_dir / 'lgbm_ranker.txt'
    full_lgb.booster_.save_model(str(lgb_path))
    logger.info(f"Full LGBMRanker saved to {lgb_path}")

    # Full XGBClassifier
    full_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        n_jobs=-1,
        verbosity=0,
    )
    full_xgb.fit(X, y)

    xgb_path = model_dir / 'xgb_ranker.json'
    full_xgb.save_model(str(xgb_path))
    logger.info(f"Full XGBClassifier saved to {xgb_path}")

    # =========================================================================
    # 6. Save meta-learner + feature names
    # =========================================================================
    meta_path = model_dir / 'stacking_meta.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'meta_model': meta_model,
            'features': features,
        }, f)
    logger.info(f"Stacking meta-model saved to {meta_path}")

    # Log feature importance from full retrained LGB
    importance = full_lgb.feature_importances_
    for i, score in enumerate(importance):
        logger.info(f"  LGB Feature {features[i]}: {score}")

    logger.info("Stacking training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train ranking models')
    parser.add_argument('--stacking', action='store_true',
                        help='Train with model stacking (LGB + XGB + Meta-Learner)')
    args = parser.parse_args()

    if args.stacking:
        train_stacking()
    else:
        train_ranker()
