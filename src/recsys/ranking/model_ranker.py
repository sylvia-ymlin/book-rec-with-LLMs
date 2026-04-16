import logging
import pickle
from pathlib import Path
from typing import List, Optional, Any

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd

from src.recsys.ranking.base import BaseRanker

logger = logging.getLogger(__name__)


class LGBMRanker(BaseRanker):
    """
    LightGBM Booster wrapper for ranking.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model: Optional[lgb.Booster] = None
        self._feature_names: List[str] = []

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            self.model = lgb.Booster(model_file=str(self.model_path))
            self._feature_names = self.model.feature_name()
            logger.info("LGBMRanker loaded from %s", self.model_path)
            return True
        except Exception as e:
            logger.error("Failed to load LGBMRanker: %s", e)
            return False

    def predict(
        self,
        user_id: str,
        candidate_items: List[str],
        features_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> np.ndarray:
        if self.model is None:
            if not self.load():
                return np.zeros(len(candidate_items))

        if features_df is None:
            logger.warning("LGBMRanker: No features_df provided for prediction")
            return np.zeros(len(candidate_items))

        # Ensure all required features are present
        X = features_df[self._feature_names]
        return self.model.predict(X)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names


class StackingRanker(BaseRanker):
    """
    Ensemble ranker using LGBM and XGBoost with a meta-model.
    """

    def __init__(
        self,
        lgbm_path: str | Path,
        xgb_path: str | Path,
        meta_path: str | Path
    ):
        self.lgbm_ranker = LGBMRanker(lgbm_path)
        self.xgb_path = Path(xgb_path)
        self.meta_path = Path(meta_path)
        self.xgb_model: Optional[Any] = None
        self.meta_model: Optional[Any] = None

    def load(self) -> bool:
        success = self.lgbm_ranker.load()
        if not success:
            return False

        # Load XGBoost
        if self.xgb_path.exists():
            try:
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(str(self.xgb_path))
            except Exception:
                try:
                    self.xgb_model = xgb.Booster()
                    self.xgb_model.load_model(str(self.xgb_path))
                except Exception as e:
                    logger.warning("StackingRanker: Failed to load XGBoost: %s", e)
                    self.xgb_model = None

        # Load Meta-model
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "rb") as f:
                    meta_data = pickle.load(f)
                    self.meta_model = meta_data["meta_model"]
                logger.info("StackingRanker: Meta-model loaded from %s", self.meta_path)
            except Exception as e:
                logger.error("StackingRanker: Failed to load meta-model: %s", e)
                return False

        return self.meta_model is not None

    def predict(
        self,
        user_id: str,
        candidate_items: List[str],
        features_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> np.ndarray:
        if self.meta_model is None:
            if not self.load():
                return self.lgbm_ranker.predict(user_id, candidate_items, features_df)

        if features_df is None:
            return np.zeros(len(candidate_items))

        # 1. Base model predictions
        lgb_scores = self.lgbm_ranker.predict(user_id, candidate_items, features_df)

        if self.xgb_model is None:
            return lgb_scores

        X = features_df[self.lgbm_ranker.feature_names]
        if isinstance(self.xgb_model, xgb.Booster):
            dtest = xgb.DMatrix(X)
            xgb_scores = self.xgb_model.predict(dtest)
        else:
            xgb_scores = self.xgb_model.predict_proba(X)[:, 1]

        # 2. Meta-model fusion
        meta_features = np.column_stack([lgb_scores, xgb_scores])
        return self.meta_model.predict_proba(meta_features)[:, 1]

    @property
    def feature_names(self) -> List[str]:
        return self.lgbm_ranker.feature_names
