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
    mrr_sum = 0.0
    
    # Cache for speed analysis
    total_time = 0
    
    results = []
    
    for idx, (_, row) in tqdm(enumerate(eval_df.iterrows()), total=len(eval_df), desc="Evaluating"):
        user_id = row['user_id']
        target_isbn = row['isbn']
        
        # Get Recs
        try:
            recs = service.get_recommendations(user_id, top_k=50) 
            
            if not recs:
                if idx < 5: 
                    logger.warning(f"Empty recs for user {user_id}")
                continue
            
            rec_isbns = [r[0] for r in recs]
            
            # Check Hit
            if target_isbn in rec_isbns:
                rank = rec_isbns.index(target_isbn)
                
                # HR@10
                if rank < 10:
                    hits += 1
                
                # MRR (consider top 50)
                # MRR@5 (Strict)
                if (rank + 1) <= 5: # Check if rank is within top 5 (1-indexed)
                    mrr_sum += 1.0 / (rank + 1)
            
        except Exception as e:
            logger.error(f"Error for user {user_id}: {e}")
            continue

    # 4. Report
    hr_10 = hits / len(eval_df)
    mean_mrr = mrr_sum / len(eval_df) # Changed from mrr to mrr_sum
    
    logger.info("==============================")
    logger.info("  EVALUATION RESULTS (Strict)") # Changed title
    logger.info("==============================")
    logger.info(f"Users Evaluated: {len(eval_df)}")
    logger.info(f"Hit Rate@10:   {hr_10:.4f}")
    logger.info(f"MRR@5:         {mean_mrr:.4f}") # Changed MRR@50 to MRR@5
    logger.info("==============================")

if __name__ == "__main__":
    evaluate_baseline(sample_n=500) # Fast check

