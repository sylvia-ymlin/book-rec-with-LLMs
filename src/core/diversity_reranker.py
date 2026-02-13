"""
Diversity Reranker: MMR + Popularity penalty + Category constraints.

ALGORITHM (MMR - Carbonell & Goldstein, 1998):
  score = lambda * relevance - (1-lambda) * max_sim(candidate, selected)
  Greedy: iteratively pick the candidate with highest MMR score.
  lambda: relevance weight. Higher = more accuracy, less diversity.

  Popularity penalty: score /= (1 + gamma * log(1 + norm_count)) to reduce
  over-recommendation of popular items (e.g. Harry Potter).

  Category constraint: max N items per category in top-k to avoid homogeneity.

P0 optimization: Improves Diversity and Serendipity without significantly
reducing Accuracy. Applied after LGBM/DIN ranking, before returning results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.infra.utils import setup_logger

logger = setup_logger(__name__)


class DiversityReranker:
    """
    Rerank candidates using MMR, popularity penalty, and category diversity.
    """

    def __init__(
        self,
        metadata_store,
        data_dir: str = "data/rec",
        mmr_lambda: float = 0.75,
        popularity_gamma: float = 0.1,
        max_per_category: int = 3,
        enable_mmr: bool = True,
        enable_popularity_penalty: bool = True,
        enable_category_constraint: bool = True,
    ):
        """
        Args:
            metadata_store: For get_book_metadata (category lookup).
            data_dir: Path to load train.csv for item popularity (interaction count).
            mmr_lambda: Relevance weight in MMR. Higher = more accuracy, less diversity.
            popularity_gamma: Penalty strength for popular items. Higher = less Harry Potter.
            max_per_category: Max items per category in top-k.
            enable_*: Feature flags.
        """
        self.metadata_store = metadata_store
        self.data_dir = Path(data_dir)
        self.mmr_lambda = mmr_lambda
        self.popularity_gamma = popularity_gamma
        self.max_per_category = max_per_category
        self.enable_mmr = enable_mmr
        self.enable_popularity_penalty = enable_popularity_penalty
        self.enable_category_constraint = enable_category_constraint

        self.item_popularity: dict = {}  # isbn -> count (interactions in train)
        self._load_item_popularity()

    def _load_item_popularity(self) -> None:
        """Load item popularity from train.csv (interaction count per ISBN)."""
        train_path = self.data_dir / "train.csv"
        if not train_path.exists():
            logger.warning("train.csv not found, popularity penalty disabled")
            return
        try:
            import pandas as pd
            df = pd.read_csv(train_path)
            if "isbn" in df.columns:
                self.item_popularity = df["isbn"].astype(str).value_counts().to_dict()
            else:
                col = [c for c in df.columns if "isbn" in c.lower()][:1]
                if col:
                    self.item_popularity = df[col[0]].astype(str).value_counts().to_dict()
            logger.info(f"DiversityReranker: Loaded popularity for {len(self.item_popularity)} items")
        except Exception as e:
            logger.warning(f"Failed to load item popularity: {e}")

    def _get_category(self, isbn: str) -> str:
        """Get item category from metadata."""
        meta = self.metadata_store.get_book_metadata(str(isbn))
        cat = meta.get("simple_categories", "") if meta else ""
        return (cat or "Unknown").strip()

    def _category_similarity(self, cat1: str, cat2: str) -> float:
        """1 if same category, 0 otherwise."""
        return 1.0 if cat1 and cat2 and cat1.lower() == cat2.lower() else 0.0

    def _get_popularity_score(self, isbn: str) -> float:
        """Log-normalized popularity (for penalty)."""
        cnt = self.item_popularity.get(str(isbn), 0)
        return float(cnt)

    def rerank(
        self,
        candidates: List[Tuple[str, float, list]],
        top_k: int,
    ) -> List[Tuple[str, float, list]]:
        """
        Rerank (isbn, score, explanations) list.

        Args:
            candidates: Sorted by score descending.
            top_k: Number of results to return.

        Returns:
            Reranked list of (isbn, score, explanations).
        """
        if not candidates:
            return []

        # 1. Popularity penalty (adjust scores before MMR)
        if self.enable_popularity_penalty:
            max_cnt = max(self._get_popularity_score(i) for i, _, _ in candidates) or 1
            adjusted = []
            for isbn, score, expl in candidates:
                cnt = self._get_popularity_score(isbn)
                # score_adj = score / (1 + gamma * log(1 + normalized_cnt))
                norm_cnt = cnt / max_cnt if max_cnt > 0 else 0
                import math
                penalty = 1.0 / (1.0 + self.popularity_gamma * math.log1p(norm_cnt * 100))
                adj_score = score * penalty
                adjusted.append((isbn, adj_score, expl))
            candidates = adjusted

        # 2. MMR rerank (diversity via category similarity)
        if self.enable_mmr and len(candidates) > 1:
            candidates = self._mmr_rerank(candidates, top_k)

        # 3. Category constraint (ensure diversity in final list)
        if self.enable_category_constraint:
            candidates = self._apply_category_constraint(candidates, top_k)
        else:
            candidates = candidates[:top_k]

        return candidates

    def _mmr_rerank(
        self,
        candidates: List[Tuple[str, float, list]],
        top_k: int,
    ) -> List[Tuple[str, float, list]]:
        """MMR: score = lambda * rel - (1-lambda) * max_sim(candidate, selected)."""
        selected: List[Tuple[str, float, list]] = []
        remaining = list(candidates)

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_mmr = float("-inf")

            for idx, (isbn, rel, expl) in enumerate(remaining):
                # Diversity: max similarity to already selected
                max_sim = 0.0
                cat_cand = self._get_category(isbn)
                for sel_isbn, _, _ in selected:
                    sim = self._category_similarity(cat_cand, self._get_category(sel_isbn))
                    max_sim = max(max_sim, sim)

                mmr = self.mmr_lambda * rel - (1.0 - self.mmr_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx < 0:
                break
            selected.append(remaining.pop(best_idx))

        return selected

    def _apply_category_constraint(
        self,
        candidates: List[Tuple[str, float, list]],
        top_k: int,
    ) -> List[Tuple[str, float, list]]:
        """Greedy: prefer items that don't exceed max_per_category."""
        category_counts: dict = {}
        result: List[Tuple[str, float, list]] = []

        for isbn, score, expl in candidates:
            if len(result) >= top_k:
                break
            cat = self._get_category(isbn)
            count = category_counts.get(cat, 0)
            if count < self.max_per_category:
                result.append((isbn, score, expl))
                category_counts[cat] = count + 1

        # If we have slack, fill with remaining (no constraint)
        if len(result) < top_k:
            seen = {r[0] for r in result}
            for isbn, score, expl in candidates:
                if len(result) >= top_k:
                    break
                if isbn not in seen:
                    result.append((isbn, score, expl))
                    seen.add(isbn)

        return result
