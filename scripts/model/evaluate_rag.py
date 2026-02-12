#!/usr/bin/env python3
"""
Evaluate RAG retrieval on a Golden Test Set.

Quantitative metrics: Accuracy@K, Recall@K, MRR@K, NDCG@K.
Use human-annotated Query-Book pairs for data-driven evaluation.

Usage:
    python scripts/model/evaluate_rag.py
    python scripts/model/evaluate_rag.py --golden data/rag_golden.csv --top_k 10

Golden set format (CSV): query, isbn, relevance
    - query: user search string
    - isbn: expected relevant book (1=relevant)
    - Multiple rows per query = multiple relevant books
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import logging
from collections import defaultdict

from src.core.recommendation_orchestrator import RecommendationOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _dcg_at_k(relevances: list[float], k: int) -> float:
    """DCG@K: sum(rel_i / log2(rank_i + 1)). relevances[i] = relevance at rank i+1."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _ndcg_at_k(relevances: list[float], k: int, n_relevant: int) -> float:
    """NDCG@K. relevances: binary (0/1) per rank. IDCG = ideal when n_relevant items at top."""
    dcg = _dcg_at_k(relevances, k)
    n_at_top = min(n_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_at_top))
    return dcg / idcg if idcg > 0 else 0.0


def load_golden(path: Path) -> dict[str, set[str]]:
    """Load golden set: {query -> set of relevant isbns}."""
    df = pd.read_csv(path, comment="#")
    df = df[df.get("relevance", 1) == 1]  # Only relevant pairs
    golden = defaultdict(set)
    for _, row in df.iterrows():
        q = str(row["query"]).strip()
        isbn = str(row["isbn"]).strip().replace(".0", "")
        if q and isbn:
            golden[q].add(isbn)
    return dict(golden)


def evaluate_rag(
    golden_path: Path | str = "data/rag_golden.csv",
    top_k: int = 10,
    use_title_match: bool = True,
) -> dict:
    """
    Run RAG retrieval on golden set and compute metrics.

    Returns: dict with accuracy_at_k, recall_at_k, mrr_at_k, ndcg_at_k, n_queries
    """
    golden_path = Path(golden_path)
    if not golden_path.exists():
        # Fallback to example
        alt = Path("data/rag_golden.example.csv")
        if alt.exists():
            logger.warning("Golden set not found at %s, using %s", golden_path, alt)
            golden_path = alt
        else:
            raise FileNotFoundError(
                f"Golden set not found. Create {golden_path} with columns: query,isbn,relevance. "
                "See data/rag_golden.example.csv for format."
            )

    golden = load_golden(golden_path)
    if not golden:
        raise ValueError("Golden set is empty")

    logger.info("Evaluating RAG on %d queries from %s", len(golden), golden_path)

    recommender = RecommendationOrchestrator()
    isbn_to_title = {}
    if use_title_match:
        try:
            bp = Path("data/books_processed.csv")
            if not bp.exists():
                bp = Path(__file__).resolve().parent.parent.parent / "data" / "books_processed.csv"
            books = pd.read_csv(bp, usecols=["isbn13", "title"])
            books["isbn13"] = books["isbn13"].astype(str).str.replace(r"\.0$", "", regex=True)
            isbn_to_title = books.set_index("isbn13")["title"].to_dict()
        except Exception as e:
            logger.warning("Could not load title map: %s", e)
            use_title_match = False

    hits_acc = 0
    recall_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for query, relevant_isbns in golden.items():
        try:
            recs = recommender.get_recommendations_sync(query, category="All")
            rec_isbns = [r.get("isbn") or r.get("isbn13") for r in recs if r]
            rec_isbns = [str(x).replace(".0", "") for x in rec_isbns if pd.notna(x)]
            rec_top = rec_isbns[:top_k]

            # Match: exact or title
            def _match(target: str, cand_list: list) -> int:
                for i, c in enumerate(cand_list):
                    if str(c).strip() == str(target).strip():
                        return i
                    if use_title_match:
                        t_title = isbn_to_title.get(str(target), "").lower().strip()
                        c_title = isbn_to_title.get(str(c), "").lower().strip()
                        if t_title and c_title and t_title == c_title:
                            return i
                return -1

            # Accuracy@K: at least one relevant in top-K
            found_any = False
            first_rank = top_k + 1
            count_in_top = 0

            for rel in relevant_isbns:
                rk = _match(rel, rec_top)
                if rk >= 0:
                    found_any = True
                    count_in_top += 1
                    first_rank = min(first_rank, rk + 1)

            if found_any:
                hits_acc += 1
            recall_sum += count_in_top / len(relevant_isbns) if relevant_isbns else 0
            if first_rank <= top_k:
                mrr_sum += 1.0 / first_rank

            # NDCG@K: build relevance vector per rank
            relevances = []
            for c in rec_top:
                matched = False
                for rel in relevant_isbns:
                    if str(c).strip() == str(rel).strip():
                        matched = True
                        break
                    if use_title_match:
                        t_title = isbn_to_title.get(str(rel), "").lower().strip()
                        c_title = isbn_to_title.get(str(c), "").lower().strip()
                        if t_title and c_title and t_title == c_title:
                            matched = True
                            break
                relevances.append(1.0 if matched else 0.0)
            ndcg_sum += _ndcg_at_k(relevances, top_k, len(relevant_isbns))

        except Exception as e:
            logger.warning("Query %r failed: %s", query[:50], e)

    n = len(golden)
    return {
        "accuracy_at_k": hits_acc / n,
        "recall_at_k": recall_sum / n,
        "mrr_at_k": mrr_sum / n,
        "ndcg_at_k": ndcg_sum / n,
        "n_queries": n,
        "top_k": top_k,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG on Golden Test Set")
    parser.add_argument("--golden", default="data/rag_golden.csv", help="Path to golden CSV")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--no-title-match", action="store_true", help="Disable relaxed title matching")
    args = parser.parse_args()

    m = evaluate_rag(
        golden_path=args.golden,
        top_k=args.top_k,
        use_title_match=not args.no_title_match,
    )

    print("\n" + "=" * 50)
    print("  RAG Golden Test Set Evaluation")
    print("=" * 50)
    print(f"  Queries:     {m['n_queries']}")
    print(f"  Top-K:       {m['top_k']}")
    print(f"  Accuracy@{m['top_k']}:  {m['accuracy_at_k']:.4f}  (any relevant in top-K)")
    print(f"  Recall@{m['top_k']}:    {m['recall_at_k']:.4f}  (fraction of relevant in top-K)")
    print(f"  MRR@{m['top_k']}:      {m['mrr_at_k']:.4f}  (mean reciprocal rank)")
    print(f"  NDCG@{m['top_k']}:     {m['ndcg_at_k']:.4f}  (normalized discounted cumulative gain)")
    print("=" * 50)


if __name__ == "__main__":
    main()
