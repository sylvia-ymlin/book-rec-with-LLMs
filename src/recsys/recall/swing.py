"""
Swing recall: item–item similarity weighted by user-pair overlap.

Moved from `src/recall/swing.py` into the `recsys.recall` package.
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Swing:
    def __init__(self, data_dir="data/rec", save_dir="data/model/recall"):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sim_matrix = {}
        self.user_hist = {}

    def fit(self, df, alpha=1.0, top_k_sim=200, max_hist=50):
        logger.info("Building Swing similarity matrix (optimized)...")

        user_items = defaultdict(set)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building index"):
            user_items[row["user_id"]].add(row["isbn"])

        self.user_hist = {u: items for u, items in user_items.items()}

        pair_users = defaultdict(list)
        for user_id, items in tqdm(
            user_items.items(), desc="Collecting item pairs"
        ):
            items_list = sorted(items)
            if len(items_list) > max_hist:
                items_list = list(
                    np.random.choice(
                        items_list, max_hist, replace=False
                    )
                )
                items_list.sort()

            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    pair_users[(items_list[i], items_list[j])].append(user_id)

        logger.info("Collected %d item pairs with shared users", len(pair_users))

        sim = defaultdict(lambda: defaultdict(float))
        for (item_i, item_j), users in tqdm(
            pair_users.items(), desc="Computing Swing"
        ):
            if len(users) < 2:
                u = users[0]
                score = 1.0 / (alpha + len(user_items[u]))
                sim[item_i][item_j] += score
                sim[item_j][item_i] += score
                continue

            u_list = users[:100]
            score = 0.0
            for idx_u in range(len(u_list)):
                u = u_list[idx_u]
                items_u = user_items[u]
                for idx_v in range(idx_u + 1, len(u_list)):
                    v = u_list[idx_v]
                    overlap = len(items_u & user_items[v])
                    score += 1.0 / (alpha + overlap)

            sim[item_i][item_j] += score
            sim[item_j][item_i] += score

        del pair_users

        logger.info("Normalizing Swing matrix...")
        final_sim = {}
        for item_i, related in tqdm(sim.items(), desc="Pruning"):
            sorted_items = sorted(
                related.items(), key=lambda x: x[1], reverse=True
            )[:top_k_sim]
            if sorted_items:
                max_score = sorted_items[0][1]
                if max_score > 0:
                    final_sim[item_i] = {
                        j: s / max_score for j, s in sorted_items
                    }

        self.sim_matrix = final_sim
        self.save()
        logger.info("Swing matrix built: %d items", len(final_sim))
        return self.sim_matrix

    def recommend(self, user_id, history_items=None, top_k=50):
        rank = defaultdict(float)

        if history_items is None:
            if user_id in self.user_hist:
                history_items = list(self.user_hist[user_id])
            else:
                return []

        history_set = set(history_items)

        for item_i in history_items:
            if item_i in self.sim_matrix:
                for item_j, score in self.sim_matrix[item_i].items():
                    if item_j in history_set:
                        continue
                    rank[item_j] += score

        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save(self):
        with open(self.save_dir / "swing.pkl", "wb") as f:
            pickle.dump(
                {"sim_matrix": self.sim_matrix, "user_hist": self.user_hist}, f
            )
        logger.info("Swing model saved to %s", self.save_dir / "swing.pkl")

    def load(self):
        path = self.save_dir / "swing.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.sim_matrix = data["sim_matrix"]
                self.user_hist = data["user_hist"]
            logger.info("Swing model loaded from %s", path)
            return True
        return False


__all__ = ["Swing"]

