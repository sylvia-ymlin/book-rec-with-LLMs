import pickle
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from pathlib import Path
from src.config import MMR_LAMBDA_DEFAULT, POPULARITY_GAMMA_DEFAULT, MAX_PER_CATEGORY_DEFAULT
from src.recsys.recall.fusion import RecallFusion
from src.recsys.ranking.features import FeatureEngineer
from src.recsys.ranking.explainer import RankingExplainer
from src.recsys.ranking.din import DINRanker
from src.core.diversity_reranker import DiversityReranker
from src.utils import setup_logger

logger = setup_logger(__name__)


class RecommendationService:
    def __init__(self, data_dir='data/rec', model_dir='data/model'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

        self.fusion = RecallFusion(data_dir, f'{model_dir}/recall')
        self.fe = FeatureEngineer(data_dir, f'{model_dir}/recall')

        self.ranker = None
        self.ranker_loaded = False
        self.din_ranker = DINRanker(str(data_dir), str(model_dir))
        self.din_ranker_loaded = False
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

        # Prefer DIN ranker when available (deep model)
        din_path = self.model_dir / 'ranking/din_ranker.pt'
        if din_path.exists():
            if self.din_ranker.load():
                self.din_ranker_loaded = True
                logger.info("DIN ranker loaded — using deep model for ranking")

        # Load LGBM ranker (fallback when DIN not available)
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
                    except Exception as e:
                        logger.debug("XGBoost Booster fallback failed: %s", e)
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

        if self.din_ranker_loaded:
            # DIN: deep model; P1: override_hist for real-time
            aux_arr = None
            if self.din_ranker.aux_feature_names:
                X_df = self.fe.generate_features_batch(
                    user_id,
                    valid_candidates,
                    override_user_emb=override_user_emb,
                    override_user_seq=effective_seq,
                )
                for col in self.din_ranker.aux_feature_names:
                    if col not in X_df.columns:
                        X_df[col] = 0
                aux_arr = X_df[self.din_ranker.aux_feature_names].values.astype(np.float32)
            scores = self.din_ranker.predict(
                user_id,
                valid_candidates,
                aux_arr,
                override_hist=effective_seq,
            )
            explanations_list = [[] for _ in valid_candidates]
            final_scores = list(zip(valid_candidates, scores, explanations_list))
            final_scores.sort(key=lambda x: x[1], reverse=True)
        elif self.ranker_loaded:
            # LGBM / stacking path. P1: override for real-time
            X_df = self.fe.generate_features_batch(
                user_id,
                valid_candidates,
                override_user_emb=override_user_emb,
                override_user_seq=effective_seq,
            )
            model_features = self.ranker.feature_name()
            for col in model_features:
                if col not in X_df.columns:
                    X_df[col] = 0
            X_df = X_df[model_features]

            if self.use_stacking and self.xgb_ranker is not None and self.meta_model is not None:
                lgb_scores = self.ranker.predict(X_df)
                if isinstance(self.xgb_ranker, xgb.Booster):
                    dtest = xgb.DMatrix(X_df)
                    xgb_scores = self.xgb_ranker.predict(dtest)
                else:
                    xgb_scores = self.xgb_ranker.predict_proba(X_df)[:, 1]
                meta_features = np.column_stack([lgb_scores, xgb_scores])
                scores = self.meta_model.predict_proba(meta_features)[:, 1]
            else:
                scores = self.ranker.predict(X_df)

            explanations_list = []
            if self.explainer is not None:
                try:
                    explanations_list = self.explainer.explain(X_df, top_k=3)
                except Exception as e:
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
