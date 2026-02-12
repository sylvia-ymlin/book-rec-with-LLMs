"""
User-based Collaborative Filtering recall.

Moved from `src/recall/usercf.py` into the `recsys.recall` package.
"""

import logging
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)


class UserCF:
    def __init__(self, data_dir="data/rec", save_dir="data/model/recall"):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.u2u_sim = {}
        self.user_hist = {}

    def fit(self, df: pd.DataFrame):
        logger.info("Building UserCF similarity matrix...")

        item_users = defaultdict(list)
        self.user_hist = defaultdict(set)

        records = df[["user_id", "isbn", "rating"]].values

        user_cnt = defaultdict(int)

        for user, item, _rating in tqdm(
            records, desc="Building Inverted Index"
        ):
            item_users[item].append(user)
            self.user_hist[user].add(item)
            user_cnt[user] += 1

        sim = defaultdict(dict)

        hot_item_limit = 200
        logger.info("Total items: %d", len(item_users))

        for item, users in tqdm(
            item_users.items(), desc="Calculating User Sim"
        ):
            if len(users) > hot_item_limit or len(users) < 2:
                continue

            weight = 1.0 / math.log(1 + len(users))

            for i, u1 in enumerate(users):
                for j, u2 in enumerate(users):
                    if i == j:
                        continue
                    sim[u1][u2] = sim[u1].get(u2, 0) + weight

        logger.info("Normalizing user similarity...")
        final_sim = {}
        for u1, related in tqdm(sim.items(), desc="Normalizing"):
            top_k_users = sorted(
                related.items(), key=lambda x: x[1], reverse=True
            )[:50]

            final_sim[u1] = {}
            for u2, score in top_k_users:
                norm = math.sqrt(user_cnt[u1] * user_cnt[u2])
                if norm > 0:
                    final_sim[u1][u2] = score / norm

        self.u2u_sim = final_sim
        self.save()
        return self.u2u_sim

    def recommend(self, user_id, history_items=None, top_k=50):
        rank = defaultdict(float)

        if user_id not in self.u2u_sim:
            return []

        similar_users = self.u2u_sim[user_id]

        seen = self.user_hist.get(user_id, set())
        if history_items:
            seen.update(history_items)

        for sim_user, similarity in similar_users.items():
            sim_user_items = self.user_hist.get(sim_user, set())
            for item in sim_user_items:
                if item in seen:
                    continue
                rank[item] += similarity

        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save(self):
        with open(self.save_dir / "usercf.pkl", "wb") as f:
            pickle.dump(
                {"u2u_sim": self.u2u_sim, "user_hist": self.user_hist}, f
            )
        logger.info("Model saved to %s", self.save_dir / "usercf.pkl")

    def load(self):
        path = self.save_dir / "usercf.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.u2u_sim = data["u2u_sim"]
                self.user_hist = data["user_hist"]
            logger.info("Model loaded from %s", path)
            return True
        return False


__all__ = ["UserCF"]

