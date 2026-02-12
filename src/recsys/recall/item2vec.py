"""
Item2Vec Recall: Word2Vec-based item embedding similarity.

Moved from `src/recall/item2vec.py` into `recsys.recall`.
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Item2Vec:
    def __init__(self, data_dir: str = "data/rec", save_dir: str = "data/model/recall"):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sim_matrix: dict[str, dict[str, float]] = {}
        self.user_hist: dict[str, set[str]] = {}

    def fit(
        self,
        df,
        vector_size: int = 64,
        window: int = 5,
        min_count: int = 3,
        sg: int = 1,
        epochs: int = 10,
        top_k_sim: int = 200,
    ):
        """
        Train Item2Vec embeddings and build similarity matrix.
        """
        logger.info("Building Item2Vec embeddings...")

        user_items: dict[str, set[str]] = defaultdict(set)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building index"):
            user_items[row["user_id"]].add(row["isbn"])
        self.user_hist = {u: items for u, items in user_items.items()}

        logger.info("Building interaction sequences...")
        df_sorted = df.sort_values(["user_id", "timestamp"])
        sentences: list[list[str]] = []
        for _user_id, group in df_sorted.groupby("user_id"):
            seq = group["isbn"].tolist()
            if len(seq) >= 2:
                sentences.append(seq)

        logger.info(
            "Built %d sequences for Word2Vec training", len(sentences)
        )

        logger.info(
            "Training Word2Vec (dim=%d, window=%d, sg=%d, epochs=%d)...",
            vector_size,
            window,
            sg,
            epochs,
        )
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=4,
            epochs=epochs,
            seed=42,
        )
        vocab_items = list(model.wv.index_to_key)
        logger.info(
            "Word2Vec trained: %d items in vocabulary", len(vocab_items)
        )

        logger.info("Building similarity matrix from embeddings...")
        final_sim: dict[str, dict[str, float]] = {}
        for item in tqdm(vocab_items, desc="Computing similarities"):
            try:
                similar = model.wv.most_similar(item, topn=top_k_sim)
                final_sim[item] = {sim_item: score for sim_item, score in similar}
            except KeyError:
                continue

        self.sim_matrix = final_sim
        self.save()
        logger.info("Item2Vec matrix built: %d items", len(final_sim))
        return self.sim_matrix

    def recommend(self, user_id, history_items=None, top_k: int = 50):
        """
        Recommend items based on embedding similarity to user history.
        Sum cosine similarity from each history item to candidate.
        """
        rank: dict[str, float] = defaultdict(float)

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
        with open(self.save_dir / "item2vec.pkl", "wb") as f:
            pickle.dump(
                {"sim_matrix": self.sim_matrix, "user_hist": self.user_hist},
                f,
            )
        logger.info(
            "Item2Vec model saved to %s", self.save_dir / "item2vec.pkl"
        )

    def load(self):
        path = self.save_dir / "item2vec.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.sim_matrix = data["sim_matrix"]
                self.user_hist = data["user_hist"]
            logger.info("Item2Vec model loaded from %s", path)
            return True
        return False


__all__ = ["Item2Vec"]

