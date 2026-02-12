"""
Item-based Collaborative Filtering with temporal direction weight.

Moved from `src/recall/itemcf.py` into the `recsys.recall` package.
"""

import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)

# Direction weights for asymmetric co-occurrence (CHANGELOG: forward=1.0, backward=0.7)
FORWARD_WEIGHT = 1.0
BACKWARD_WEIGHT = 0.7


class ItemCF:
    """
    Item-based Collaborative Filtering with Temporal Direction Weight.

    See original `src/recall/itemcf.py` for detailed documentation.
    """

    def __init__(self, data_dir: str = "data/rec", save_dir: str = "data/model/recall"):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir.parent / "recall_models.db"
        self.conn: Optional[sqlite3.Connection] = None

    def fit(self, df: pd.DataFrame, top_k_sim: int = 200) -> "ItemCF":
        logger.info(
            "Building ItemCF similarity matrix (direction-weighted co-occurrence)..."
        )

        user_seqs: dict[str, list[tuple[str, float]]] = defaultdict(list)
        if "timestamp" in df.columns:
            for _, row in tqdm(
                df.iterrows(), total=len(df), desc="Building user sequences"
            ):
                user_seqs[row["user_id"]].append((str(row["isbn"]), float(row["timestamp"])))
            for uid in user_seqs:
                user_seqs[uid] = [
                    x[0] for x in sorted(user_seqs[uid], key=lambda t: t[1])
                ]
        else:
            for _, row in tqdm(
                df.iterrows(), total=len(df), desc="Building user sequences"
            ):
                user_seqs[row["user_id"]].append(str(row["isbn"]))

        user_hist = {u: list(items) for u, items in user_seqs.items()}

        item_users: dict[str, set[str]] = defaultdict(set)
        for user_id, items in user_hist.items():
            for item in items:
                item_users[item].add(user_id)
        item_counts = {k: len(v) for k, v in item_users.items()}

        sim: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for _user_id, items in tqdm(user_hist.items(), desc="Computing co-occurrence"):
            for i in range(len(items)):
                item_i = items[i]
                for j in range(i + 1, len(items)):
                    item_j = items[j]
                    sim[item_i][item_j] += FORWARD_WEIGHT
                    sim[item_j][item_i] += BACKWARD_WEIGHT

        logger.info("Normalizing ItemCF matrix...")
        final_sim: dict[str, dict[str, float]] = {}
        for item_i, related in tqdm(sim.items(), desc="Normalizing"):
            ni = item_counts.get(item_i, 1)
            pruned = sorted(
                related.items(), key=lambda x: x[1], reverse=True
            )[:top_k_sim]
            final_sim[item_i] = {}
            for item_j, raw_score in pruned:
                nj = item_counts.get(item_j, 1)
                norm = math.sqrt(ni * nj)
                if norm > 0:
                    final_sim[item_i][item_j] = raw_score / norm

        self._sim_matrix = final_sim
        self._user_hist = user_hist
        self.save()
        logger.info("ItemCF built: %d items, saved to %s", len(final_sim), self.db_path)
        return self

    def save(self) -> None:
        if not hasattr(self, "_sim_matrix") or not hasattr(self, "_user_hist"):
            logger.warning("ItemCF.save: No fitted model to save.")
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS item_similarity")
        cursor.execute(
            """
            CREATE TABLE item_similarity (item1 TEXT, item2 TEXT, score REAL)
            """
        )
        cursor.execute("DROP TABLE IF EXISTS user_history")
        cursor.execute(
            """
            CREATE TABLE user_history (user_id TEXT, isbn TEXT)
            """
        )

        batch = []
        for item1, related in tqdm(
            self._sim_matrix.items(), desc="Writing item_similarity"
        ):
            for item2, score in related.items():
                batch.append((item1, item2, score))
                if len(batch) >= 100000:
                    cursor.executemany(
                        "INSERT INTO item_similarity VALUES (?, ?, ?)", batch
                    )
                    batch = []
        if batch:
            cursor.executemany(
                "INSERT INTO item_similarity VALUES (?, ?, ?)", batch
            )

        batch = []
        for user_id, isbns in tqdm(
            self._user_hist.items(), desc="Writing user_history"
        ):
            for isbn in isbns:
                batch.append((user_id, isbn))
                if len(batch) >= 100000:
                    cursor.executemany(
                        "INSERT INTO user_history VALUES (?, ?)", batch
                    )
                    batch = []
        if batch:
            cursor.executemany(
                "INSERT INTO user_history VALUES (?, ?)", batch
            )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_item1 ON item_similarity(item1)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_user ON user_history(user_id)"
        )
        conn.commit()
        conn.close()
        logger.info("ItemCF saved to %s", self.db_path)

    def load(self) -> bool:
        if self.db_path.exists():
            try:
                self.conn = sqlite3.connect(
                    str(self.db_path), check_same_thread=False
                )
                logger.info("ItemCF: Connected to SQLite %s", self.db_path)
                return True
            except Exception as e:
                logger.error("ItemCF: Failed to connect to SQLite: %s", e)
        else:
            logger.warning("ItemCF: SQLite DB not found at %s", self.db_path)
        return False

    def recommend(self, user_id, history_items=None, top_k=50):
        """Recommend items for a user based on history from SQLite."""
        if not self.conn and not self.load():
            return []

        cursor = self.conn.cursor()

        if history_items is None:
            cursor.execute(
                "SELECT isbn FROM user_history WHERE user_id = ?", (user_id,)
            )
            history_items = [row[0] for row in cursor.fetchall()]

        if not history_items:
            return []

        from collections import defaultdict as dd

        rank = dd(float)

        placeholders = ", ".join(["?"] * len(history_items))
        query = f"""
            SELECT item2, SUM(score) as total_score
            FROM item_similarity
            WHERE item1 IN ({placeholders})
            AND item2 NOT IN ({placeholders})
            GROUP BY item2
            ORDER BY total_score DESC
            LIMIT ?
        """

        try:
            params = history_items + history_items + [top_k]
            cursor.execute(query, params)
            results = cursor.fetchall()
            return results
        except Exception as e:
            logger.error("ItemCF Query Error: %s", e)
            return []


__all__ = ["ItemCF", "FORWARD_WEIGHT", "BACKWARD_WEIGHT"]

