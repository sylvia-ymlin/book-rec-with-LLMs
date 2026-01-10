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

        # Load ISBN -> Title map for deduplication
        try:
            books_df = pd.read_csv('data/books_processed.csv', usecols=['isbn13', 'title'])
            # Ensure isbn13 is str
            books_df['isbn13'] = books_df['isbn13'].astype(str).str.replace(r'\.0$', '', regex=True)
            self.isbn_to_title = pd.Series(
                books_df.title.values, 
                index=books_df.isbn13.values
            ).to_dict()
            logger.info("Loaded ISBN-Title map for deduplication.")
        except Exception as e:
            logger.warning(f"Could not load books for deduplication: {e}")
            self.isbn_to_title = {}
            
    def get_recommendations(self, user_id, top_k=10):
        """
        Get personalized recommendations for a user
        """
        from src.user.profile_store import list_favorites
        
        self.load_resources()
        
        # 0. Get User Context (Favorites) for filtering
        try:
            user_favs = list_favorites(user_id)
            # list_favorites returns ['isbn1', 'isbn2']
            fav_isbns = set(user_favs)
        except Exception as e:
            logger.warning(f"Could not fetch favorites for filtering: {e}")
            fav_isbns = set()

        # 1. Recall
        # Get ~100 candidates (oversample to allow for filtering)
        candidates = self.fusion.get_recall_items(user_id, k=150)
        if not candidates:
            return []
            
        candidate_items = [item for item, score in candidates]
        
        # 2. Ranking
        if self.ranker_loaded:
            # Generate features
            feats_list = []
            valid_candidates = []
            
            for item in candidate_items:
                # Filter 1: Already in favorites
                if item in fav_isbns:
                    continue
                valid_candidates.append(item)
                f = self.fe.generate_features(user_id, item)
                feats_list.append(f)
            
            if not valid_candidates:
                return []
                
            X_df = pd.DataFrame(feats_list)
            
            # Filter features to match model
            if hasattr(self.ranker, 'feature_names_in_'):
                valid_features = self.ranker.feature_names_in_
                # Align columns
                # Missing cols filled with 0? Or XGB handles it?
                # Usually better to ensure columns exist
                for col in valid_features:
                    if col not in X_df.columns:
                        X_df[col] = 0
                X_df = X_df[valid_features]
            
            # Predict
            scores = self.ranker.predict_proba(X_df)[:, 1]
            
            # Combine
            final_scores = list(zip(valid_candidates, scores))
            final_scores.sort(key=lambda x: x[1], reverse=True)
            
        else:
            # Fallback to recall scores, but filter
            final_scores = []
            for item, score in candidates:
                if item not in fav_isbns:
                    final_scores.append((item, score))
        
        # 3. Deduplication by Title
        unique_results = []
        seen_titles = set()
        
        # Ensure map exists (fallback)
        if not hasattr(self, 'isbn_to_title'):
            self.isbn_to_title = {}
        
        for isbn, score in final_scores:
            title = self.isbn_to_title.get(str(isbn), "").lower().strip()
            
            # If title is found and seen, skip
            if title and title in seen_titles:
                continue
                
            if title:
                seen_titles.add(title)
            
            unique_results.append((isbn, score))
            if len(unique_results) >= top_k:
                break
                
        return unique_results

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
