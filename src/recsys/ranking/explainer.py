"""
SHAP-based ranking explainer.

Moved from `src/ranking/explainer.py` into `recsys.ranking`.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import shap


logger = logging.getLogger(__name__)


FEATURE_LABELS = {
    "u_cnt": "Reading Volume",
    "u_mean": "Your Avg Rating",
    "u_std": "Rating Diversity",
    "i_cnt": "Book Popularity",
    "i_mean": "Book Avg Rating",
    "i_std": "Rating Controversy",
    "len_diff": "Complexity Match",
    "u_auth_avg": "Author Rating",
    "u_auth_match": "Known Author",
    "sasrec_score": "Reading Pattern",
    "sim_max": "Similar to Recent",
    "sim_min": "Diversity Score",
    "sim_mean": "Recent Fit",
    "is_cat_hob": "Category Match",
    "icf_sum": "Similar Books",
    "icf_max": "Best Book Match",
    "ucf_sum": "Reader Community",
}


class RankingExplainer:
    """
    Wraps a SHAP TreeExplainer around the LGBMRanker.
    """

    def __init__(self, lgbm_booster):
        self.explainer = shap.TreeExplainer(lgbm_booster)
        logger.info("SHAP TreeExplainer initialized for LGBMRanker")

    def explain(self, X_df: pd.DataFrame, top_k: int = 3) -> List[List[Dict]]:
        """
        Compute SHAP values for all rows in X_df and return top-k contributing
        features per row.
        """
        shap_values = self.explainer.shap_values(X_df)

        feature_names = list(X_df.columns)
        explanations: List[List[Dict]] = []

        for i in range(len(X_df)):
            row_shap = shap_values[i]

            abs_contribs = np.abs(row_shap)
            top_indices = np.argsort(abs_contribs)[::-1][:top_k]

            row_explanation: List[Dict] = []
            for idx in top_indices:
                feat_name = feature_names[idx]
                shap_val = float(row_shap[idx])

                if abs(shap_val) < 1e-6:
                    continue

                row_explanation.append(
                    {
                        "feature": FEATURE_LABELS.get(feat_name, feat_name),
                        "contribution": round(shap_val, 4),
                        "direction": "positive" if shap_val > 0 else "negative",
                    }
                )

            explanations.append(row_explanation)

        return explanations


__all__ = ["RankingExplainer"]

