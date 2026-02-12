"""
Swing Recall: item-item similarity weighted by user-pair overlap.

ALGORITHM (Zhou et al., 2008-style):
  For each pair of users (u, v) who both interacted with items i and j:
      swing(i, j) += 1 / (alpha + |I_u ∩ I_v|)

  Intuition: Small overlap (|I_u ∩ I_v|) → high weight. Users with very similar
  taste (large overlap) contribute less distinctive signal, so we down-weight them.
  This reduces popularity bias compared to raw co-occurrence.

COMPLEXITY:
  Optimized: iterate users → enumerate item pairs per user → O(users × items²)
  Avoids the naive O(items² × users²) by not iterating over all item pairs first.

PARAMETERS:
  alpha: Smoothing. Higher = more penalty on overlap (default 1.0).
  top_k_sim: Keep only top-k similar items per item to reduce memory.
  max_hist: Cap user history length; very active users have noisy signal.
"""

import pickle
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class Swing:
    def __init__(self, data_dir='data/rec', save_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sim_matrix = {}
        self.user_hist = {}

    def fit(self, df, alpha=1.0, top_k_sim=200, max_hist=50):
        """
        Build Swing similarity matrix.

        Optimized approach: iterate users, enumerate item pairs from each user's
        history, accumulate co-occurring user lists per item pair, then compute
        swing scores from user-pair overlaps.

        Args:
            df: DataFrame with [user_id, isbn, rating, timestamp]
            alpha: smoothing factor (higher = more penalty on overlap)
            top_k_sim: keep only top-k similar items per item
            max_hist: cap user history length (skip very active users)
        """
        logger.info("Building Swing similarity matrix (optimized)...")

        # 1. Build user -> items mapping
        user_items = defaultdict(set)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building index"):
            user_items[row['user_id']].add(row['isbn'])

        self.user_hist = {u: items for u, items in user_items.items()}

        # 2. For each item pair, collect users who interacted with BOTH items.
        # Key: (item_i, item_j) canonical order (i < j). Value: list of user_ids.
        # This enables computing |I_u ∩ I_v| per pair in step 3.
        pair_users = defaultdict(list)

        for user_id, items in tqdm(user_items.items(), desc="Collecting item pairs"):
            items_list = sorted(items)  # canonical order
            # Skip users with too many items (noisy signal)
            if len(items_list) > max_hist:
                items_list = list(np.random.choice(items_list, max_hist, replace=False))
                items_list.sort()

            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    pair_users[(items_list[i], items_list[j])].append(user_id)

        logger.info(f"Collected {len(pair_users)} item pairs with shared users")

        # 3. Compute Swing score: sum over user pairs of 1/(alpha + overlap).
        sim = defaultdict(lambda: defaultdict(float))

        for (item_i, item_j), users in tqdm(pair_users.items(), desc="Computing Swing"):
            if len(users) < 2:
                # Single user: no pair, use 1/(alpha + |I_u|) as weight.
                u = users[0]
                score = 1.0 / (alpha + len(user_items[u]))
                sim[item_i][item_j] += score
                sim[item_j][item_i] += score
                continue

            # Cap user pairs for very popular item pairs (avoid O(n²) explosion).
            u_list = users[:100]

            # Swing: sum over all (u,v) pairs of 1/(alpha + |I_u ∩ I_v|).
            score = 0.0
            for idx_u in range(len(u_list)):
                u = u_list[idx_u]
                items_u = user_items[u]
                for idx_v in range(idx_u + 1, len(u_list)):
                    v = u_list[idx_v]
                    overlap = len(items_u & user_items[v])  # |I_u ∩ I_v|
                    score += 1.0 / (alpha + overlap)

            sim[item_i][item_j] += score
            sim[item_j][item_i] += score

        del pair_users  # free memory

        # 4. Normalize and keep top-k per item
        logger.info("Normalizing Swing matrix...")
        final_sim = {}
        for item_i, related in tqdm(sim.items(), desc="Pruning"):
            sorted_items = sorted(related.items(), key=lambda x: x[1], reverse=True)[:top_k_sim]
            if sorted_items:
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
