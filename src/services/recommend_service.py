import logging
import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
from src.recall.fusion import RecallFusion
from src.ranking.features import FeatureEngineer

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, data_dir='data/rec', model_dir='data/model'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        
        self.fusion = RecallFusion(data_dir, f'{model_dir}/recall')
        self.fe = FeatureEngineer(data_dir, f'{model_dir}/recall')
        
        self.ranker = None
        self.ranker_loaded = False
        
    def load_resources(self):
        if self.ranker_loaded:
            return
            
        logger.info("Loading Recommendation Service resources...")
        self.fusion.load_models()
        self.fe.load_base_data()
        
        # Load Ranker
        ranker_path = self.model_dir / 'ranking/xgb_ranker.json'
        if ranker_path.exists():
            self.ranker = xgb.XGBClassifier()
            self.ranker.load_model(ranker_path)
            logger.info(f"Ranker loaded from {ranker_path}")
            self.ranker_loaded = True
        else:
            logger.warning(f"Ranker model not found at {ranker_path}, prediction will be skipped")
            
    def get_recommendations(self, user_id, top_k=10):
        """
        Get personalized recommendations for a user
        """
        self.load_resources()
        
        # 1. Recall
        # Get ~100 candidates
        candidates = self.fusion.get_recall_items(user_id, k=100)
        if not candidates:
            return []
            
        candidate_items = [item for item, score in candidates]
        
        # 2. Ranking
        if self.ranker_loaded:
            # Generate features
            feats_list = []
            for item in candidate_items:
                f = self.fe.generate_features(user_id, item)
                feats_list.append(f)
            
            X_df = pd.DataFrame(feats_list)
            
            # Predict
            # output is probability of class 1
            scores = self.ranker.predict_proba(X_df)[:, 1]
            
            # Combine with recall score? 
            # Or just use ranker score? Usually ranker score is enough.
            final_scores = list(zip(candidate_items, scores))
            final_scores.sort(key=lambda x: x[1], reverse=True)
            
            return final_scores[:top_k]
        else:
            # Fallback to recall scores
            return candidates[:top_k]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = RecommendationService()
    
    # Test user
    df = pd.read_csv('data/rec/train.csv')
    user_id = df['user_id'].iloc[0]
    
    logger.info(f"Getting recommendations for {user_id}...")
    recs = service.get_recommendations(user_id)
    
    print("\nTop Recommendations:")
    for item, score in recs:
        print(f"ISBN: {item}, Score: {score:.4f}")
