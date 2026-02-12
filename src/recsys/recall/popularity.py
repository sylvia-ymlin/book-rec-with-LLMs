"""
Popularity-based global recall.

Moved from `src/recall/popularity.py` into `recsys.recall`.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class PopularityRecall:
    def __init__(self, data_dir: str = "data/rec", save_dir: str = "data/model/recall"):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.hot_items: list[str] = []

    def fit(self, df: pd.DataFrame):
        """
        Calculate popularity score from interactions.
        """
        logger.info("Calculating popularity...")

        stats = df.groupby("isbn").agg(
            {"user_id": "count", "rating": "mean"}
        ).rename(columns={"user_id": "count", "rating": "avg_rating"})

        stats["score"] = stats["count"]

        self.hot_items = stats.sort_values("score", ascending=False).index.tolist()

        self.save()
        return self.hot_items

    def recommend(self, user_id=None, top_k: int = 50):
        """
        Return top-K globally popular items.
        """
        return [
            (item, 1.0 / (i + 1))
            for i, item in enumerate(self.hot_items[:top_k])
        ]

    def save(self):
        with open(self.save_dir / "popularity.pkl", "wb") as f:
            pickle.dump(self.hot_items, f)
        logger.info("Model saved to %s", self.save_dir / "popularity.pkl")

    def load(self):
        path = self.save_dir / "popularity.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.hot_items = pickle.load(f)
            logger.info("Model loaded from %s", path)
            return True
        return False


__all__ = ["PopularityRecall"]

