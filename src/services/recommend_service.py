import logging
import pickle
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from pathlib import Path
from src.recall.fusion import RecallFusion
from src.ranking.features import FeatureEngineer
from src.ranking.explainer import RankingExplainer

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, data_dir='data/rec', model_dir='data/model'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

        self.fusion = RecallFusion(data_dir, f'{model_dir}/recall')
        self.fe = FeatureEngineer(data_dir, f'{model_dir}/recall')

        self.ranker = None
        self.ranker_loaded = False
        self.xgb_ranker = None
        self.meta_model = None
        self.use_stacking = False
        self.explainer = None  # SHAP explainer (V2.7)

    def load_resources(self):
        if self.ranker_loaded:
            return

        logger.info("Loading Recommendation Service resources...")
        self.fusion.load_models()
        self.fe.load_base_data()

        # Load Ranker (LightGBM)
        ranker_path = self.model_dir / 'ranking/lgbm_ranker.txt'
        if ranker_path.exists():
            self.ranker = lgb.Booster(model_file=str(ranker_path))
            logger.info(f"Ranker loaded from {ranker_path}")
            self.ranker_loaded = True

            # Initialize SHAP explainer (V2.7)
            try:
                self.explainer = RankingExplainer(self.ranker)
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")
                self.explainer = None

            # Load XGBoost ranker (for stacking)
            xgb_path = self.model_dir / 'ranking/xgb_ranker.json'
            if xgb_path.exists():
                try:
                    self.xgb_ranker = xgb.XGBClassifier()
                    # For older models/new xgboost versions, loading might raise TypeError if type isn't set
                    self.xgb_ranker.load_model(str(xgb_path))
                    logger.info(f"XGBoost ranker loaded from {xgb_path}")
                except Exception as e:
                    logger.warning(f"Failed to load XGBoost ranker (stacking might be suboptimal): {e}")
                    # Fallback to booster if it's a raw booster dump
                    try:
                        self.xgb_ranker = xgb.Booster()
                        self.xgb_ranker.load_model(str(xgb_path))
                        logger.info("XGBoost ranker loaded as raw Booster.")
                    except:
                        self.xgb_ranker = None

            # Load stacking meta-model
            meta_path = self.model_dir / 'ranking/stacking_meta.pkl'
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    meta_data = pickle.load(f)
                    self.meta_model = meta_data['meta_model']
                self.use_stacking = True
                logger.info(f"Stacking meta-model loaded — stacking ENABLED")
        else:
            logger.warning(f"Ranker model not found at {ranker_path}, prediction will be skipped")

        # Deduplication now uses MetadataStore for Title lookups (Zero-RAM mode)
        from src.core.metadata_store import metadata_store
        self.metadata_store = metadata_store
        logger.info("RecommendationService: Zero-RAM mode enabled for metadata lookups.")

    def get_recommendations(self, user_id, top_k=10, filter_favorites=True):
        """
        Get personalized recommendations for a user.

        Returns:
            List of (isbn, score, explanations) tuples where explanations
            is a list of dicts with feature contributions from SHAP.
        """
        from src.user.profile_store import list_favorites

        self.load_resources()

        # 0. Get User Context (Favorites) for filtering
        fav_isbns = set()
        if filter_favorites:
            try:
                user_favs = list_favorites(user_id)
                fav_isbns = set(user_favs)
            except Exception as e:
                logger.warning(f"Could not fetch favorites for filtering: {e}")

        # 1. Recall
        # Get candidates (oversample to allow for filtering)
        candidates = self.fusion.get_recall_items(user_id, k=200)
        if not candidates:
            return []

        # Deduplicate candidates (keep highest score)
        unique_candidates = {}
        for item, score in candidates:
            if item not in unique_candidates:
                unique_candidates[item] = score

        candidates = list(unique_candidates.items())
        candidate_items = [item for item, score in candidates]

        # 2. Ranking
        if self.ranker_loaded:
            # Filter candidates first
            valid_candidates = [item for item in candidate_items if item not in fav_isbns]
            
            if not valid_candidates:
                return []

            # Batch Feature Generation (Optimized)
            X_df = self.fe.generate_features_batch(user_id, valid_candidates)

            # Align features to match model
            model_features = self.ranker.feature_name()
            for col in model_features:
                if col not in X_df.columns:
                    X_df[col] = 0
            X_df = X_df[model_features]

            # Predict
            if self.use_stacking and self.xgb_ranker is not None and self.meta_model is not None:
                # Stacking: Level-1 predictions -> Level-2 meta-learner
                lgb_scores = self.ranker.predict(X_df)
                
                # Check if XGB Ranker is a raw Booster or Sklearn Estimator
                if isinstance(self.xgb_ranker, xgb.Booster):
                    dtest = xgb.DMatrix(X_df)
                    xgb_scores = self.xgb_ranker.predict(dtest)
                else:
                    xgb_scores = self.xgb_ranker.predict_proba(X_df)[:, 1]
                meta_features = np.column_stack([lgb_scores, xgb_scores])
                scores = self.meta_model.predict_proba(meta_features)[:, 1]
            else:
                # Fallback: LightGBM only (backward compatible)
                scores = self.ranker.predict(X_df)

            # Compute SHAP explanations (V2.7)
            explanations_list = []
            if self.explainer is not None:
                try:
                    explanations_list = self.explainer.explain(X_df, top_k=3)
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
                    explanations_list = [[] for _ in valid_candidates]
            else:
                explanations_list = [[] for _ in valid_candidates]

            # Combine with explanations
            final_scores = list(zip(valid_candidates, scores, explanations_list))
            final_scores.sort(key=lambda x: x[1], reverse=True)

        else:
            # Fallback to recall scores, but filter
            final_scores = []
            for item, score in candidates:
                if item not in fav_isbns:
                    final_scores.append((item, score, []))

        # 3. Deduplication by Title
        unique_results = []
        seen_titles = set()

        for isbn, score, explanation in final_scores:
            meta = self.metadata_store.get_book_metadata(str(isbn))
            title = meta.get("title", "").lower().strip() if meta else ""

            # If title is found and seen, skip
            if title and title in seen_titles:
                continue

            if title:
                seen_titles.add(title)

            unique_results.append((isbn, score, explanation))
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
    for item, score, explanation in recs:
        print(f"ISBN: {item}, Score: {score:.4f}")
        for exp in explanation:
            print(f"  → {exp['feature']}: {exp['contribution']:+.4f} ({exp['direction']})")
