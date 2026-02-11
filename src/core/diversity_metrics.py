"""
P3: Diversity evaluation metrics.

ILSD (Intra-List Similarity Diversity), Category Coverage, Gini.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


def category_coverage(
    rec_isbns: List[str],
    get_category: Callable[[str], str],
    top_k: int = 10,
) -> float:
    """
    Fraction of unique categories in top-k list.
    Higher = more diverse.
    """
    if not rec_isbns or top_k <= 0:
        return 0.0
    rec_top = rec_isbns[:top_k]
    cats = {get_category(isbn) for isbn in rec_top}
    cats.discard("")
    cats.discard("Unknown")
    return len(cats) / max(len(rec_top), 1)


def intra_list_similarity(
    rec_isbns: List[str],
    similarity_fn: Callable[[str, str], float],
    top_k: int = 10,
) -> float:
    """
    Average pairwise similarity within top-k.
    Lower = more diverse. ILSD = 1 - this (when similarity in [0,1]).
    """
    if not rec_isbns or top_k <= 0:
        return 0.0
    rec_top = rec_isbns[:top_k]
    n = len(rec_top)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += similarity_fn(rec_top[i], rec_top[j])
            count += 1
    return total / count if count > 0 else 0.0


def category_coverage_similarity(isbn1: str, isbn2: str, get_category: Callable[[str], str]) -> float:
    """1 if same category, 0 otherwise. Used for ILSD proxy."""
    return 1.0 if get_category(isbn1) == get_category(isbn2) else 0.0


def compute_diversity_metrics(
    rec_isbns: List[str],
    get_category: Callable[[str], str],
    top_k: int = 10,
) -> dict:
    """
    Compute category coverage and category-based ILSD.
    Returns dict with category_coverage, ilsd (1 - avg_category_sim).
    """
    cov = category_coverage(rec_isbns, get_category, top_k)
    sim_fn = lambda a, b: category_coverage_similarity(a, b, get_category)
    sim = intra_list_similarity(rec_isbns, sim_fn, top_k)
    return {
        "category_coverage": cov,
        "ilsd": 1.0 - sim,  # higher = more diverse
    }
