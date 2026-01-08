import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from src.services.recommend_service import RecommendationService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_baseline(sample_n=1000):
    logger.info("Initializing Evaluation...")
    
    # 1. Load Test Data
    test_df = pd.read_csv('data/rec/test.csv')
    
    # Sample users
    if sample_n and sample_n < len(test_df):
        eval_df = test_df.sample(n=sample_n, random_state=42)
    else:
        eval_df = test_df
        
    logger.info(f"Evaluating on {len(eval_df)} users...")
    
    # 2. Init Service
    service = RecommendationService()
    service.load_resources()
    
    # 3. Predict & Metric
    k = 10
    hits = 0
    mrr = 0.0
    
    # Cache for speed analysis
    total_time = 0
    
    results = []
    
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
        user_id = row['user_id']
        target_isbn = row['isbn']
        
        # Get Recs
        try:
            # We need higher recall from service to check rank
            recs = service.get_recommendations(user_id, top_k=50) # Get top 50
            rec_isbns = [r[0] for r in recs]
            
            # Check Hit
            if target_isbn in rec_isbns:
                rank = rec_isbns.index(target_isbn)
                
                # HR@10
                if rank < 10:
                    hits += 1
                
                # MRR (consider top 50)
                mrr += 1.0 / (rank + 1)
            
        except Exception as e:
            logger.error(f"Error for user {user_id}: {e}")
            continue

    # 4. Report
    hr_10 = hits / len(eval_df)
    mean_mrr = mrr / len(eval_df)
    
    logger.info("="*30)
    logger.info("  BASELINE EVALUATION RESULTS")
    logger.info("="*30)
    logger.info(f"Users Evaluated: {len(eval_df)}")
    logger.info(f"Hit Rate@10:   {hr_10:.4f}")
    logger.info(f"MRR@50:        {mean_mrr:.4f}")
    logger.info("="*30)

if __name__ == "__main__":
    evaluate_baseline(sample_n=500) # Fast check
