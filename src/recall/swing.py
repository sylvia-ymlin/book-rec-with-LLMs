import pickle
import math
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Swing:
    """
    Swing recall: item-item similarity weighted by user-pair overlap.

    For each pair of users (u, v) who both interacted with items i and j:
        swing(i, j) += 1 / (alpha + |I_u ∩ I_v|)

    This penalizes user pairs with large overlap (less distinctive signal).
    """

    def __init__(self, data_dir='data/rec', save_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sim_matrix = {}
        self.user_hist = {}

    def fit(self, df, alpha=1.0, max_users_per_item=500, top_k_sim=200):
        """
        Build Swing similarity matrix.

        Args:
            df: DataFrame with [user_id, isbn, rating, timestamp]
            alpha: smoothing factor (higher = more penalty on overlap)
            max_users_per_item: cap users per item to control compute
            top_k_sim: keep only top-k similar items per item
        """
        logger.info("Building Swing similarity matrix...")

        # 1. Build inverted index: item -> set of users
        item_users = defaultdict(set)
        user_items = defaultdict(set)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building index"):
            u, i = row['user_id'], row['isbn']
            item_users[i].add(u)
            user_items[u].add(i)

        self.user_hist = {u: items for u, items in user_items.items()}

        # 2. Prune: cap users per item for speed
        for item in item_users:
            users = item_users[item]
            if len(users) > max_users_per_item:
                item_users[item] = set(list(users)[:max_users_per_item])

        # 3. Compute Swing similarity
        # For each item, find co-occurring items via shared users
        sim = defaultdict(lambda: defaultdict(float))
        items = list(item_users.keys())

        for item_i in tqdm(items, desc="Computing Swing"):
            users_i = item_users[item_i]

            # Collect co-occurring items through users of item_i
            cooccur_items = defaultdict(list)  # item_j -> list of users who have both
            for u in users_i:
                for item_j in user_items[u]:
                    if item_j != item_i:
                        cooccur_items[item_j].append(u)

            # For each co-occurring item, compute swing score
            for item_j, shared_users in cooccur_items.items():
                if len(shared_users) < 2:
                    # Need at least 2 users for a user pair
                    # Single user co-occurrence is handled by ItemCF
                    score = 0.0
                    for u in shared_users:
                        score += 1.0 / (alpha + len(user_items[u]))
                    sim[item_i][item_j] += score
                    continue

                # Swing: iterate user pairs
                users_list = shared_users[:50]  # cap pairs for speed
                for idx_u in range(len(users_list)):
                    u = users_list[idx_u]
                    for idx_v in range(idx_u + 1, len(users_list)):
                        v = users_list[idx_v]
                        overlap = len(user_items[u] & user_items[v])
                        swing_score = 1.0 / (alpha + overlap)
                        sim[item_i][item_j] += swing_score

        # 4. Normalize and keep top-k
        logger.info("Normalizing Swing matrix...")
        final_sim = {}
        for item_i, related in tqdm(sim.items(), desc="Pruning"):
            # Sort by score and keep top_k
            sorted_items = sorted(related.items(), key=lambda x: x[1], reverse=True)[:top_k_sim]
            if sorted_items:
                # Normalize by max score for this item
                max_score = sorted_items[0][1]
                if max_score > 0:
                    final_sim[item_i] = {j: s / max_score for j, s in sorted_items}

        self.sim_matrix = final_sim
        self.save()
        logger.info(f"Swing matrix built: {len(final_sim)} items")
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
        with open(self.save_dir / 'swing.pkl', 'wb') as f:
            pickle.dump({
                'sim_matrix': self.sim_matrix,
                'user_hist': self.user_hist
            }, f)
        logger.info(f"Swing model saved to {self.save_dir / 'swing.pkl'}")

    def load(self):
        path = self.save_dir / 'swing.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.sim_matrix = data['sim_matrix']
                self.user_hist = data['user_hist']
            logger.info(f"Swing model loaded from {path}")
            return True
        return False
