"""
SHAP-based Ranking Explainer (V2.7)

Computes per-candidate feature contributions using TreeExplainer
on the LGBMRanker, then maps raw feature names to human-readable labels.

Usage:
    explainer = RankingExplainer(lgbm_booster)
    explanations = explainer.explain(X_df, top_k=3)
    # explanations[i] = [
    #   {"feature": "Known Author", "contribution": 0.42, "direction": "positive"},
    #   ...
    # ]
"""

import logging
import shap
import numpy as np
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

# Human-readable labels for each ranking feature
FEATURE_LABELS = {
    "u_cnt":         "Reading Volume",
    "u_mean":        "Your Avg Rating",
    "u_std":         "Rating Diversity",
    "i_cnt":         "Book Popularity",
    "i_mean":        "Book Avg Rating",
    "i_std":         "Rating Controversy",
    "len_diff":      "Complexity Match",
    "u_auth_avg":    "Author Rating",
    "u_auth_match":  "Known Author",
    "sasrec_score":  "Reading Pattern",
    "sim_max":       "Similar to Recent",
    "sim_min":       "Diversity Score",
    "sim_mean":      "Recent Fit",
    "is_cat_hob":    "Category Match",
    "icf_sum":       "Similar Books",
    "icf_max":       "Best Book Match",
    "ucf_sum":       "Reader Community",
}


class RankingExplainer:
    """
    Wraps a SHAP TreeExplainer around the LGBMRanker.

    Uses TreeExplainer (exact, fast for tree ensembles) to compute
    per-sample SHAP values, then returns the top-k contributing
    features with human-readable labels.
    """

    def __init__(self, lgbm_booster):
        """
        Args:
            lgbm_booster: A lightgbm.Booster loaded from lgbm_ranker.txt
        """
        self.explainer = shap.TreeExplainer(lgbm_booster)
        logger.info("SHAP TreeExplainer initialized for LGBMRanker")

    def explain(self, X_df: pd.DataFrame, top_k: int = 3) -> List[List[Dict]]:
        """
        Compute SHAP values for all rows in X_df and return
        top-k contributing features per row.

        Args:
            X_df: DataFrame with shape (n_candidates, 17 features)
                  columns must match the LGBMRanker's feature names
            top_k: number of top contributing features to return per candidate

        Returns:
            List of length n_candidates, where each element is a list of dicts:
            [
                {"feature": "Known Author", "contribution": 0.42, "direction": "positive"},
                {"feature": "Reading Pattern", "contribution": 0.31, "direction": "positive"},
                ...
            ]
        """
        # shap_values shape: (n_samples, n_features)
        shap_values = self.explainer.shap_values(X_df)

        feature_names = list(X_df.columns)
        explanations = []

        for i in range(len(X_df)):
            row_shap = shap_values[i]  # (n_features,)

            # Sort by absolute contribution descending
            abs_contribs = np.abs(row_shap)
            top_indices = np.argsort(abs_contribs)[::-1][:top_k]

            row_explanation = []
            for idx in top_indices:
                feat_name = feature_names[idx]
                shap_val = float(row_shap[idx])

                # Skip near-zero contributions
                if abs(shap_val) < 1e-6:
                    continue

                row_explanation.append({
                    "feature": FEATURE_LABELS.get(feat_name, feat_name),
                    "contribution": round(shap_val, 4),
                    "direction": "positive" if shap_val > 0 else "negative",
                })

            explanations.append(row_explanation)

        return explanations
