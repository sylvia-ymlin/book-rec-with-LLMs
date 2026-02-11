import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import pickle
from pathlib import Path
from src.ranking.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_ranker_data(data_dir='data/rec', neg_ratio=4):
    """
    Construct training data for ranker
    Positives: from val.csv
    Negatives: extracted from random sampling (ignoring user history)
    """
    logger.info("Building ranker training data...")
    val_df = pd.read_csv(f'{data_dir}/val.csv')
    
    # 1. Positives
    pos_samples = val_df[['user_id', 'isbn']].copy()
    pos_samples['label'] = 1
    
    # 2. Negatives
    # Get all items
    all_items = pd.read_csv(f'{data_dir}/train.csv')['isbn'].unique()
    
    neg_samples = []
    # Vectorized sampling is faster
    # Assuming we sample neg_ratio negatives per user
    users = pos_samples['user_id'].values
    n_neg = len(users) * neg_ratio
    
    sampled_items = np.random.choice(all_items, size=n_neg)
    
    # Repeat users
    sampled_users = np.repeat(users, neg_ratio)
    
    neg_df = pd.DataFrame({
        'user_id': sampled_users,
        'isbn': sampled_items,
        'label': 0
    })
    
    # Ideally checking against history, but for speed we ignore collision (rare)
    
    train_data = pd.concat([pos_samples, neg_df], ignore_index=True)
    # Shuffle
    train_data = train_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return train_data

def train_ranker():
    data_dir = Path('data/rec')
    model_dir = Path('data/model/ranking')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare Data
    train_samples = build_ranker_data(str(data_dir))
    logger.info(f"Training samples: {len(train_samples)}")
    
    # 2. Generate Features
    # Feature Engineer needs recall models loaded
    fe = FeatureEngineer(data_dir=str(data_dir), model_dir='data/model/recall')
    
    logger.info("Generating features (this may take a while)...")
    # For speed, maybe subset?
    # Full set ~840k. 
    # generate_features involves dict lookups. 840k * 0.1ms = 84s. Fast.
    X_y = fe.create_dateset(train_samples)
    
    # 3. Train XGBoost
    features = [c for c in X_y.columns if c not in ['label', 'user_id', 'isbn']]
    target = 'label'
    
    X = X_y[features]
    y = X_y[target]
    
    logger.info(f"Training XGBoost with features: {features}")
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    
    # 4. Save
    model_path = model_dir / 'xgb_ranker.json'
    model.save_model(model_path)
    logger.info(f"Ranker saved to {model_path}")
    
    # Feature Importance
    importance = model.feature_importances_
    for i, score in enumerate(importance):
        logger.info(f"Feature {features[i]}: {score:.4f}")

if __name__ == "__main__":
    train_ranker()
