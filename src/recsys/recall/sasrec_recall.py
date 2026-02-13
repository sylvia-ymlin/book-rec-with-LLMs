"""
SASRec-based recall: sequence embedding + Faiss ANN search.

Moved from `src/recall/sasrec_recall.py` into the `recsys.recall` package.
"""

import logging
import pickle
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from src.recsys.recall.sequence_utils import build_sequences_from_df


logger = logging.getLogger(__name__)


class SASRecRecall:
    """
    Lightweight sequence recall with SASRec-compatible interface.

    This project is a research prototype, so we keep a robust fallback path:
    if deep SASRec artifacts are unavailable, we still provide stable
    sequence-based recall (recent-items boost + popularity backfill).
    """

    def __init__(self, data_dir="data/rec", model_dir="data/model/recall"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

        self.user_seq_emb: dict[str, np.ndarray] = {}
        self.item_emb: Optional[np.ndarray] = None
        self.item_map: dict[str, int] = {}
        self.id_to_item: dict[int, str] = {}
        self.user_hist: dict[str, list[int]] = {}
        self.user_sequences: dict[str, list[int]] = {}
        self.faiss_index = None
        self.popular_item_ids: list[int] = []
        self.loaded = False
        self._max_len = 50

    def _build_popularity(self) -> None:
        counter: Counter[int] = Counter()
        for seq in self.user_sequences.values():
            counter.update(i for i in seq if i > 0)
        self.popular_item_ids = [iid for iid, _ in counter.most_common()]

    def fit(self, df: pd.DataFrame) -> None:
        """Build sequence artifacts from interactions and persist to disk."""
        if df is None or df.empty:
            logger.warning("SASRecRecall.fit received empty dataframe")
            return

        self.user_sequences, self.item_map = build_sequences_from_df(df, max_len=self._max_len)
        self.id_to_item = {v: k for k, v in self.item_map.items()}
        self.user_hist = self.user_sequences

        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.data_dir / "user_sequences.pkl", "wb") as f:
            pickle.dump(self.user_sequences, f)
        with open(self.data_dir / "item_map.pkl", "wb") as f:
            pickle.dump(self.item_map, f)

        # Keep a placeholder user embedding artifact for compatibility.
        # Real deep embeddings are optional in this prototype.
        self.user_seq_emb = {}
        with open(self.data_dir / "user_seq_emb.pkl", "wb") as f:
            pickle.dump(self.user_seq_emb, f)

        self._build_popularity()
        self.loaded = True

    def load(self) -> bool:
        """Load prebuilt sequence artifacts; fallback gracefully when missing."""
        try:
            seq_path = self.data_dir / "user_sequences.pkl"
            map_path = self.data_dir / "item_map.pkl"
            emb_path = self.data_dir / "user_seq_emb.pkl"

            if seq_path.exists():
                with open(seq_path, "rb") as f:
                    self.user_sequences = pickle.load(f)
            else:
                self.user_sequences = {}

            if map_path.exists():
                with open(map_path, "rb") as f:
                    self.item_map = pickle.load(f)
            else:
                self.item_map = {}

            self.id_to_item = {v: k for k, v in self.item_map.items()}
            self.user_hist = self.user_sequences

            if emb_path.exists():
                with open(emb_path, "rb") as f:
                    self.user_seq_emb = pickle.load(f)
            else:
                self.user_seq_emb = {}

            self._build_popularity()
            self.loaded = True
            return True
        except Exception as e:
            logger.error("Failed to load SASRec artifacts: %s", e)
            self.user_sequences = {}
            self.item_map = {}
            self.id_to_item = {}
            self.user_hist = {}
            self.user_seq_emb = {}
            self.popular_item_ids = []
            self.loaded = False
            return False

    def _compute_emb_from_seq(self, seq_isbns: list[str]) -> Optional[np.ndarray]:
        """
        Return a lightweight sequence embedding for downstream feature hooks.
        """
        if not seq_isbns:
            return None
        ids = [self.item_map.get(str(isbn), 0) for isbn in seq_isbns]
        ids = [i for i in ids if i > 0]
        if not ids:
            return None
        arr = np.array(ids[-self._max_len:], dtype=np.float32)
        # Simple normalized signal vector (prototype-friendly, deterministic).
        return arr / (np.linalg.norm(arr) + 1e-8)

    def recommend(
        self,
        user_id,
        history_items=None,  # noqa: ARG002
        top_k: int = 50,
        real_time_seq=None,
    ):
        """
        Sequence-based recall:
        1) recent viewed/read items boost
        2) popularity backfill for coverage
        """
        if not self.loaded:
            self.load()

        # Determine sequence source
        seq_ids: list[int] = []
        if real_time_seq:
            seq_ids = [
                self.item_map.get(str(isbn), 0) for isbn in real_time_seq
            ]
            seq_ids = [i for i in seq_ids if i > 0]
        if not seq_ids:
            seq_ids = self.user_sequences.get(str(user_id), [])

        scores: dict[int, float] = {}
        seen = set(seq_ids)

        # Recent items get higher weights (reverse chronological)
        for rank, iid in enumerate(reversed(seq_ids[-self._max_len:]), start=1):
            if iid <= 0:
                continue
            scores[iid] = max(scores.get(iid, 0.0), 1.0 / rank)

        # Popularity backfill
        for iid in self.popular_item_ids:
            if len(scores) >= top_k * 3:
                break
            if iid in seen:
                continue
            # Low but non-zero score so personalized recency still dominates
            scores.setdefault(iid, 0.01)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recs = []
        for iid, sc in ranked:
            isbn = self.id_to_item.get(iid)
            if not isbn:
                continue
            recs.append((isbn, float(sc)))
            if len(recs) >= top_k:
                break
        return recs


__all__ = ["SASRecRecall"]

