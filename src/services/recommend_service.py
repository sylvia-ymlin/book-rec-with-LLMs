import pickle
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from pathlib import Path
from src.infra.config import MMR_LAMBDA_DEFAULT, POPULARITY_GAMMA_DEFAULT, MAX_PER_CATEGORY_DEFAULT
from src.recsys.recall.fusion import RecallFusion
from src.recsys.ranking.features import FeatureEngineer
from src.recsys.ranking.explainer import RankingExplainer
from src.recsys.ranking.din import DINRanker
from src.recsys.ranking.model_ranker import BaseRanker, LGBMRanker, StackingRanker
from src.core.diversity_reranker import DiversityReranker
from src.infra.utils import setup_logger

logger = setup_logger(__name__)


class RecommendationService:
    def __init__(self, data_dir='data/rec', model_dir='data/model'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

        self.fusion = RecallFusion(data_dir, f'{model_dir}/recall')
        self.fe = FeatureEngineer(data_dir, f'{model_dir}/recall')

        self.ranker: Optional[BaseRanker] = None
        self.ranker_loaded = False
        self.explainer = None  # SHAP explainer (V2.7)

    def load_resources(self):
        if self.ranker_loaded:
            return

        logger.info("Loading Recommendation Service resources...")
        self.fusion.load_models()
        self.fe.load_base_data()

        # 1. Try to load DIN ranker (highest priority)
        din_path = self.model_dir / 'ranking/din_ranker.pt'
        if din_path.exists():
            din = DINRanker(str(self.data_dir), str(self.model_dir.parent))
            if din.load():
                self.ranker = din
                self.ranker_loaded = True
                logger.info("Unified Ranker: DIN loaded")

        # 2. Try to load Stacking ranker (second priority)
        if not self.ranker_loaded:
            lgbm_path = self.model_dir / 'ranking/lgbm_ranker.txt'
            xgb_path = self.model_dir / 'ranking/xgb_ranker.json'
            meta_path = self.model_dir / 'ranking/stacking_meta.pkl'
            if lgbm_path.exists() and meta_path.exists():
                stacker = StackingRanker(lgbm_path, xgb_path, meta_path)
                if stacker.load():
                    self.ranker = stacker
                    self.ranker_loaded = True
                    logger.info("Unified Ranker: Stacking loaded")

        # 3. Try to load plain LGBM ranker (lowest priority)
        if not self.ranker_loaded:
            lgbm_path = self.model_dir / 'ranking/lgbm_ranker.txt'
            if lgbm_path.exists():
                lgbm = LGBMRanker(lgbm_path)
                if lgbm.load():
                    self.ranker = lgbm
                    self.ranker_loaded = True
                    logger.info("Unified Ranker: LGBM loaded")

        if not self.ranker_loaded:
            logger.warning("No ranking model found. Predictions will be skipped.")

        # 4. Initialize SHAP explainer if LGBM is available (it's the base for both LGBM and Stacking)
        if self.ranker_loaded:
            base_lgbm_booster = None
            if isinstance(self.ranker, LGBMRanker):
                base_lgbm_booster = self.ranker.model
            elif isinstance(self.ranker, StackingRanker):
                base_lgbm_booster = self.ranker.lgbm_ranker.model

            if base_lgbm_booster:
                try:
                    self.explainer = RankingExplainer(base_lgbm_booster)
                except Exception as e:
                    logger.warning(f"Failed to initialize SHAP explainer: {e}")
                    self.explainer = None

        # Deduplication now uses MetadataStore for Title lookups (Zero-RAM mode)
        from src.data.stores.metadata_store import metadata_store
        self.metadata_store = metadata_store
        logger.info("RecommendationService: Zero-RAM mode enabled for metadata lookups.")

        # P0: Diversity Reranker (MMR + Popularity penalty + Category constraint)
        self.diversity_reranker = DiversityReranker(
            metadata_store=metadata_store,
            data_dir=str(self.data_dir),
            mmr_lambda=MMR_LAMBDA_DEFAULT,
            popularity_gamma=POPULARITY_GAMMA_DEFAULT,
            max_per_category=MAX_PER_CATEGORY_DEFAULT,
        )

    def get_recommendations(
        self,
        user_id,
        top_k=10,
        filter_favorites=True,
        enable_diversity_rerank: bool = True,
        real_time_sequence=None,
    ):
        """
        Get personalized recommendations for a user.

        Args:
            enable_diversity_rerank: If True, apply MMR + popularity penalty + category
                diversity (P0 optimization). Can disable for A/B testing.
            real_time_sequence: P1 - List of ISBNs from current session (e.g. just-clicked).
                Injected into SASRec recall and DIN/LGBM ranking.

        Returns:
            List of (isbn, score, explanations) tuples where explanations
            is a list of dicts with feature contributions from SHAP.
        """
        from src.data.stores.profile_store import list_favorites

        self.load_resources()

        # P1: Build effective sequence (offline + real-time) for SASRec/DIN
        effective_seq = None
        override_user_emb = None
        if real_time_sequence:
            sasrec = self.fusion.sasrec
            base = getattr(sasrec, "user_sequences", {}).get(user_id, [])
            id2item = getattr(sasrec, "id_to_item", {})
            base_isbns = [id2item[i] for i in base if i in id2item]
            effective_seq = (base_isbns + list(real_time_sequence))[-50:]
            try:
                override_user_emb = sasrec._compute_emb_from_seq(effective_seq)
            except Exception:
                override_user_emb = None

        # 0. Get User Context (Favorites) for filtering
        fav_isbns = set()
        if filter_favorites:
            try:
                from src.data.stores.profile_store import list_favorites as _list_favorites

                user_favs = _list_favorites(user_id)
                fav_isbns = set(user_favs)
            except Exception as e:
                logger.warning(f"Could not fetch favorites for filtering: {e}")

        # 1. Recall (P1: inject real_time_seq into SASRec)
        candidates = self.fusion.get_recall_items(
            user_id, k=200, real_time_seq=real_time_sequence
        )
        # P1: Cold-start fallback — when recall returns empty, use popularity
        if not candidates:
            pop_recs = self.fusion.popularity.recommend(user_id, top_k=200)
            candidates = list(pop_recs)
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
        valid_candidates = [item for item in candidate_items if item not in fav_isbns]
        if not valid_candidates:
            return []

        if self.ranker_loaded:
            # Generate features for ranker
            X_df = self.fe.generate_features_batch(
                user_id,
                valid_candidates,
                override_user_emb=override_user_emb,
                override_user_seq=effective_seq,
            )

            # Predict using unified interface
            scores = self.ranker.predict(
                user_id,
                valid_candidates,
                features_df=X_df,
                override_hist=effective_seq,
                override_user_emb=override_user_emb,
            )

            # Explanations (only for LGBM-based models)
            explanations_list = []
            if self.explainer is not None:
                try:
                    # Explainer needs the subset of features the model was trained on
                    model_features = self.ranker.feature_name() if hasattr(self.ranker, "feature_name") else self.ranker.feature_names
                    # Handle if ranker.feature_name is method or property
                    if callable(model_features):
                        model_features = model_features()
                    
                    X_expl = X_df[model_features]
                    explanations_list = self.explainer.explain(X_expl, top_k=3)
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
                    explanations_list = [[] for _ in valid_candidates]
            else:
                explanations_list = [[] for _ in valid_candidates]

            final_scores = list(zip(valid_candidates, scores, explanations_list))
            final_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            # Fallback to recall scores, but filter
            final_scores = []
            for item, score in candidates:
                if item not in fav_isbns:
                    final_scores.append((item, score, []))

        # 2.5 P0: Diversity Rerank (MMR + popularity penalty + category constraint)
        if enable_diversity_rerank and final_scores:
            final_scores = self.diversity_reranker.rerank(
                final_scores,
                top_k=top_k * 2,  # Oversample for title dedup
            )

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

    def get_popular_books(self, limit: int = 24) -> list:
        """
        P2: Return popular books for onboarding selection.
        Used when new user has no history — lets them pick 3–5 to seed preferences.
        """
        self.load_resources()
        recs = self.fusion.popularity.recommend(user_id=None, top_k=limit)
        results = []
        seen_titles = set()
        for isbn, _ in recs:
            meta = self.metadata_store.get_book_metadata(str(isbn))
            title = (meta.get("title") or "").lower().strip()
            if title and title in seen_titles:
                continue
            if title:
                seen_titles.add(title)
            results.append((isbn, meta or {}))
            if len(results) >= limit:
                break
        return results

if __name__ == "__main__":
    import logging
    logger.setLevel(logging.INFO)
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
