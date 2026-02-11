"""
Shared utilities for building user sequences from interaction DataFrames.
Used by SASRec and YoutubeDNN training.
"""

from typing import Tuple

import pandas as pd
from tqdm import tqdm


def build_sequences_from_df(
    df: pd.DataFrame, max_len: int = 50
) -> Tuple[dict[str, list[int]], dict[str, int]]:
    """
    Build user sequences and item map from interaction DataFrame.

    Args:
        df: DataFrame with columns [user_id, isbn] and optionally [timestamp].
        max_len: Maximum sequence length (truncate from the left).

    Returns:
        user_seqs: Dict[user_id, list of item_ids] (1-indexed, 0 is padding)
        item_map: Dict[isbn, item_id]
    """
    items = df["isbn"].astype(str).unique()
    item_map = {isbn: i + 1 for i, isbn in enumerate(items)}

    user_history: dict[str, list[tuple[str, float]]] = {}
    has_ts = "timestamp" in df.columns

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building sequences"):
        u = str(row["user_id"])
        isbn = str(row["isbn"])
        ts = float(row["timestamp"]) if has_ts else 0.0
        if u not in user_history:
            user_history[u] = []
        user_history[u].append((isbn, ts))

    user_seqs: dict[str, list[int]] = {}
    for u, pairs in user_history.items():
        if has_ts:
            pairs.sort(key=lambda x: x[1])
        item_ids = [item_map.get(isbn, 0) for isbn, _ in pairs]
        item_ids = [x for x in item_ids if x != 0]
        user_seqs[u] = item_ids[-max_len:]

    return user_seqs, item_map
