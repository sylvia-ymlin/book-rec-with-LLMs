"""
Feature engineering for ranking.

Moved from `src/ranking/features.py` into `recsys.ranking`.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, data_dir: str = "data/rec", model_dir: str = "data/model/recall"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.recall_fusion = None

        self.user_stats: dict = {}
        self.item_stats: dict = {}
        self.has_sasrec: bool = False

    def _load_sasrec_features(self):
        """Load pre-trained SASRec embeddings."""
        logger.info("Loading SASRec Embeddings...")
        try:
            import pickle
            import torch

            with open(self.data_dir / "user_seq_emb.pkl", "rb") as f:
                self.user_seq_emb = pickle.load(f)

            with open(self.data_dir / "item_map.pkl", "rb") as f:
                self.sasrec_item_map = pickle.load(f)

            self.sasrec_model_path = (
                self.model_dir.parent / "rec" / "sasrec_model.pth"
            )
            state_dict = torch.load(self.sasrec_model_path, map_location="cpu")
            self.sas_item_emb = state_dict["item_emb.weight"].numpy()

            self.has_sasrec = True
            logger.info("SASRec features loaded.")
        except Exception as e:
            logger.warning("Failed to load SASRec features: %s", e)
            self.has_sasrec = False

    def _load_user_sequences(self):
        """Load user reading sequences (ordered by time) for Last-N similarity."""
        logger.info("Loading user sequences for Last-N similarity...")
        try:
            import pickle

            with open(self.data_dir / "user_sequences.pkl", "rb") as f:
                self.user_sequences = pickle.load(f)
            logger.info("Loaded sequences for %d users", len(self.user_sequences))
        except Exception as e:
            logger.warning("Failed to load user sequences: %s", e)
            self.user_sequences = {}

    def load_base_data(self):
        """Load feature maps via MetadataStore singleton."""
        logger.info("Accessing MetadataStore for ranking features...")
        from src.data.stores.metadata_store import metadata_store

        self.user_stats = metadata_store.user_stats
        self.item_stats = metadata_store.item_stats
        self.item_category = metadata_store.item_category
        self.user_cat_prefs = {}
        self.item_desc_len = {}
        self.item_author = metadata_store.item_author
        self.user_author_stats = {}
        self.user_avg_desc_len = {}

        logger.info("FeatureEngineer: Linked to MetadataStore maps.")

        from src.recsys.recall.fusion import RecallFusion

        self.recall_fusion = RecallFusion(self.data_dir, self.model_dir)
        self.recall_fusion.load_models()

        self._load_sasrec_features()
        self._load_user_sequences()

    def generate_features(
        self,
        user_id,
        candidate_item,
        override_user_emb=None,
        override_user_seq=None,
    ):
        """
        Generate feature vector for a (user, item) pair.
        """
        feats: dict[str, float] = {}

        u_stat = self.user_stats.get(user_id, {"count": 0, "mean": 0, "std": 0})
        feats["u_cnt"] = np.log1p(u_stat["count"])
        feats["u_mean"] = u_stat["mean"]
        feats["u_std"] = u_stat["std"] if not pd.isna(u_stat["std"]) else 0

        i_stat = self.item_stats.get(
            candidate_item, {"count": 0, "mean": 0, "std": 0}
        )
        feats["i_cnt"] = np.log1p(i_stat["count"])
        feats["i_mean"] = i_stat["mean"]
        feats["i_std"] = i_stat["std"] if not pd.isna(i_stat["std"]) else 0

        u_desc_len = self.user_avg_desc_len.get(user_id, 500)
        i_desc_len = self.item_desc_len.get(candidate_item, 0)
        feats["len_diff"] = abs(u_desc_len - i_desc_len)

        author = self.item_author.get(candidate_item, "Unknown")
        if (user_id, author) in self.user_author_stats:
            feats["u_auth_avg"] = self.user_author_stats[(user_id, author)]
            feats["u_auth_match"] = 1
        else:
            feats["u_auth_avg"] = feats["u_mean"]
            feats["u_auth_match"] = 0

        if self.has_sasrec:
            u_emb = (
                override_user_emb
                if override_user_emb is not None
                else self.user_seq_emb.get(user_id, None)
            )

            i_idx = self.sasrec_item_map.get(candidate_item, 0)

            sas_score = 0.0
            if u_emb is not None and i_idx > 0:
                i_emb = self.sas_item_emb[i_idx]
                sas_score = float(np.dot(u_emb, i_emb))

            feats["sasrec_score"] = sas_score
        else:
            feats["sasrec_score"] = 0.0

        sim_max, sim_min, sim_mean = 0.0, 0.0, 0.0
        user_seq = None
        if override_user_seq is not None and self.has_sasrec:
            user_seq = [
                self.sasrec_item_map.get(str(i), 0) for i in override_user_seq
            ]
            user_seq = [x for x in user_seq if x > 0][-5:]
        elif self.has_sasrec and hasattr(self, "user_sequences"):
            user_seq = self.user_sequences.get(user_id, [])
        if self.has_sasrec and user_seq:
            i_idx = self.sasrec_item_map.get(candidate_item, 0)
            if len(user_seq) > 0 and i_idx > 0:
                cand_emb = self.sas_item_emb[i_idx]
                last_n_indices = user_seq[-5:]

                sims: list[float] = []
                for hist_idx in last_n_indices:
                    if hist_idx > 0 and hist_idx < len(self.sas_item_emb):
                        hist_emb = self.sas_item_emb[hist_idx]
                        norm_cand = np.linalg.norm(cand_emb)
                        norm_hist = np.linalg.norm(hist_emb)
                        if norm_cand > 0 and norm_hist > 0:
                            sim = float(
                                np.dot(cand_emb, hist_emb) / (norm_cand * norm_hist)
                            )
                            sims.append(sim)

                if sims:
                    sim_max = max(sims)
                    sim_min = min(sims)
                    sim_mean = np.mean(sims)

        feats["sim_max"] = sim_max
        feats["sim_min"] = sim_min
        feats["sim_mean"] = sim_mean

        is_cat_hob = 0
        if hasattr(self, "item_category") and hasattr(self, "user_cat_prefs"):
            cand_cat = self.item_category.get(candidate_item, "Unknown")
            user_cats = self.user_cat_prefs.get(user_id, set())
            if cand_cat in user_cats:
                is_cat_hob = 1
        feats["is_cat_hob"] = is_cat_hob

        icf_score = 0.0
        icf_max = 0.0

        itemcf = self.recall_fusion.itemcf
        usercf = self.recall_fusion.usercf

        history = set()
        if hasattr(usercf, "user_hist"):
            history = usercf.user_hist.get(user_id, set())

        if hasattr(itemcf, "sim_matrix") and candidate_item in itemcf.sim_matrix:
            related = itemcf.sim_matrix[candidate_item]
            sims: list[float] = []
            for hist_item in history:
                if hist_item in related:
                    sims.append(related[hist_item])

            if sims:
                icf_score = sum(sims)
                icf_max = max(sims)
        feats["icf_sum"] = icf_score
        feats["icf_max"] = icf_max

        ucf_score = 0.0
        if hasattr(usercf, "u2u_sim") and user_id in usercf.u2u_sim:
            sim_users = usercf.u2u_sim[user_id]
            for sim_u, sim_score in sim_users.items():
                if hasattr(usercf, "user_hist") and candidate_item in usercf.user_hist.get(
                    sim_u, set()
                ):
                    ucf_score += sim_score
        feats["ucf_sum"] = ucf_score

        return feats

    def generate_features_batch(
        self,
        user_id,
        candidate_items,
        override_user_emb=None,
        override_user_seq=None,
    ):
        """
        Optimized batch feature generation for a single user and multiple items.
        """
        u_stat = self.user_stats.get(user_id, {"count": 0, "mean": 0, "std": 0})
        u_cnt = np.log1p(u_stat["count"])
        u_mean = u_stat["mean"]
        u_std = u_stat["std"] if not pd.isna(u_stat["std"]) else 0

        u_desc_len = self.user_avg_desc_len.get(user_id, 500)
        user_cats = self.user_cat_prefs.get(user_id, set())

        usercf = self.recall_fusion.usercf

        history = set()
        if hasattr(usercf, "user_hist"):
            history = usercf.user_hist.get(user_id, set())

        usercf_sim_users = {}
        if hasattr(usercf, "u2u_sim") and user_id in usercf.u2u_sim:
            usercf_sim_users = usercf.u2u_sim[user_id]

        sasrec_scores = np.zeros(len(candidate_items))
        has_sas = False
        if self.has_sasrec:
            u_emb = (
                override_user_emb
                if override_user_emb is not None
                else self.user_seq_emb.get(user_id, None)
            )
            if u_emb is not None:
                indices = [
                    self.sasrec_item_map.get(item, 0) for item in candidate_items
                ]
                idx_array = np.array(indices)
                target_embs = self.sas_item_emb[idx_array]
                scores = target_embs @ u_emb
                sasrec_scores = scores
                has_sas = True

        data = []
        itemcf = self.recall_fusion.itemcf

        for idx, item in enumerate(candidate_items):
            row: dict[str, float] = {}

            row["u_cnt"] = u_cnt
            row["u_mean"] = u_mean
            row["u_std"] = u_std

            i_stat = self.item_stats.get(
                item, {"count": 0, "mean": 0, "std": 0}
            )
            row["i_cnt"] = np.log1p(i_stat["count"])
            row["i_mean"] = i_stat["mean"]
            row["i_std"] = i_stat["std"] if not pd.isna(i_stat["std"]) else 0

            i_desc_len = self.item_desc_len.get(item, 0)
            row["len_diff"] = abs(u_desc_len - i_desc_len)

            author = self.item_author.get(item, "Unknown")
            if (user_id, author) in self.user_author_stats:
                row["u_auth_avg"] = self.user_author_stats[(user_id, author)]
                row["u_auth_match"] = 1
            else:
                row["u_auth_avg"] = u_mean
                row["u_auth_match"] = 0

            row["sasrec_score"] = float(sasrec_scores[idx]) if has_sas else 0.0

            sim_max, sim_min, sim_mean = 0.0, 0.0, 0.0
            user_seq = None
            if override_user_seq is not None and self.has_sasrec:
                user_seq = [
                    self.sasrec_item_map.get(str(i), 0) for i in override_user_seq
                ]
                user_seq = [x for x in user_seq if x > 0][-5:]
            elif hasattr(self, "user_sequences"):
                user_seq = self.user_sequences.get(user_id, [])[-5:]
            if has_sas and user_seq:
                i_idx_map = self.sasrec_item_map.get(item, 0)
                if len(user_seq) > 0 and i_idx_map > 0:
                    # reuse single-item path for correctness
                    feats_single = self.generate_features(
                        user_id,
                        item,
                        override_user_emb=override_user_emb,
                        override_user_seq=override_user_seq,
                    )
                    sim_max = feats_single.get("sim_max", 0.0)
                    sim_min = feats_single.get("sim_min", 0.0)
                    sim_mean = feats_single.get("sim_mean", 0.0)

            row["sim_max"] = sim_max
            row["sim_min"] = sim_min
            row["sim_mean"] = sim_mean

            cand_cat = self.item_category.get(item, "Unknown")
            row["is_cat_hob"] = 1 if cand_cat in user_cats else 0

            icf_score = 0.0
            icf_max = 0.0

            if hasattr(itemcf, "sim_matrix") and item in itemcf.sim_matrix:
                related = itemcf.sim_matrix[item]
                common = history.intersection(related.keys())
                if common:
                    sims = [related[c] for c in common]
                    icf_score = sum(sims)
                    icf_max = max(sims)

            row["icf_sum"] = icf_score
            row["icf_max"] = icf_max

            ucfscore = 0.0
            for sim_u, sim_score in usercf_sim_users.items():
                if hasattr(usercf, "user_hist") and item in usercf.user_hist.get(
                    sim_u, set()
                ):
                    ucfscore += sim_score
            row["ucf_sum"] = ucfscore

            data.append(row)

        return pd.DataFrame(data)

    def create_dateset(self, df_samples: pd.DataFrame):
        """
        Create DataFrame with features.
        """
        if not self.user_stats:
            self.load_base_data()

        data = []

        for _, row in tqdm(
            df_samples.iterrows(),
            total=len(df_samples),
            desc="Generating Features",
        ):
            user = row["user_id"]
            item = row["isbn"]
            label = row.get("label", 0)

            f = self.generate_features(user, item)
            f["label"] = label
            data.append(f)

        return pd.DataFrame(data)


__all__ = ["FeatureEngineer"]

