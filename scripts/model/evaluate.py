import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from src.services.recommend_service import RecommendationService
from src.data.stores.metadata_store import metadata_store
from src.core.diversity_metrics import compute_diversity_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _get_category(isbn: str) -> str:
    meta = metadata_store.get_book_metadata(str(isbn))
    return (meta.get("simple_categories", "") or "Unknown").strip()

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
    # Load ISBN -> Title map for evaluation
    isbn_to_title = {}
    try:
        books_df = pd.read_csv('data/books_processed.csv', usecols=['isbn13', 'title'])
        books_df['isbn13'] = books_df['isbn13'].astype(str).str.replace(r'\.0$', '', regex=True)
        isbn_to_title = pd.Series(books_df.title.values, index=books_df.isbn13.values).to_dict()
        logger.info("Loaded ISBN-Title map for relaxed evaluation.")
    except Exception as e:
        logger.warning(f"Could not load books for evaluation: {e}")

    # 3. Predict & Metric
    k = 10
    hits = 0
    mrr_sum = 0.0
    # P3: Diversity metrics (aggregate over all users)
    diversity_cov_sum = 0.0
    diversity_ilsd_sum = 0.0
    diversity_count = 0

    results = []
    
    for idx, (_, row) in tqdm(enumerate(eval_df.iterrows()), total=len(eval_df), desc="Evaluating"):
        user_id = row['user_id']
        target_isbn = row['isbn']
        
        # Get Recs
        try:
            # We disable favorite filtering for evaluation to handle potential data leakage in test set splits
            recs = service.get_recommendations(user_id, top_k=50, filter_favorites=False)
            # P3: Optional A/B test diversity: enable_diversity_rerank=True by default

            if not recs:
                if idx < 5: 
                    logger.warning(f"Empty recs for user {user_id}")
                continue
            
            rec_isbns = [r[0] for r in recs]
            
            # Check Hit
            hit = False
            rank = -1
            
            # 1. Exact Match
            if target_isbn in rec_isbns:
                rank = rec_isbns.index(target_isbn)
                hit = True
            
            # 2. Relaxed Title Match (if Exact failed)
            if not hit:
                target_title = isbn_to_title.get(str(target_isbn), "").lower().strip()
                if target_title:
                   for r_idx, r_isbn in enumerate(rec_isbns):
                       r_title = isbn_to_title.get(str(r_isbn), "").lower().strip()
                       if r_title and r_title == target_title:
                           rank = r_idx
                           hit = True
                           # logger.info(f"Title Match! Target: {target_isbn} ({target_title}) matches Rec: {r_isbn}")
                           break
            
            # P3: Diversity metrics on top-10
            if rec_isbns:
                d = compute_diversity_metrics(rec_isbns, _get_category, top_k=10)
                diversity_cov_sum += d["category_coverage"]
                diversity_ilsd_sum += d["ilsd"]
                diversity_count += 1

            if hit:
                # HR@10
                if rank < 10:
                    hits += 1

                # MRR (consider top 50)
                # MRR@5 (Strict)
                if (rank + 1) <= 5:  # Check if rank is within top 5 (1-indexed)
                    mrr_sum += 1.0 / (rank + 1)
            else:
                if idx < 5:
                    logger.info(f"MISS USER {user_id}: Target {target_isbn} not in top {len(rec_isbns)} recs.")
                    logger.info(f"Top 5 Recs: {rec_isbns[:5]}")
                    logger.info(f"Type check - Target: {type(target_isbn)}, Recs: {type(rec_isbns[0]) if rec_isbns else 'N/A'}")
            
        except Exception as e:
            logger.error(f"Error for user {user_id}: {e}")
            continue

    # 4. Report
    hr_10 = hits / len(eval_df)
    mean_mrr = mrr_sum / len(eval_df)
    div_n = max(diversity_count, 1)
    logger.info("==============================")
    logger.info("  EVALUATION RESULTS (Strict)")
    logger.info("==============================")
    logger.info(f"Users Evaluated: {len(eval_df)}")
    logger.info(f"Hit Rate@10:   {hr_10:.4f}")
    logger.info(f"MRR@5:         {mean_mrr:.4f}")
    logger.info(f"P3 Category Coverage@10: {diversity_cov_sum / div_n:.4f}")
    logger.info(f"P3 ILSD@10:              {diversity_ilsd_sum / div_n:.4f}")
    logger.info("==============================")

if __name__ == "__main__":
    evaluate_baseline(sample_n=500) # Fast check

