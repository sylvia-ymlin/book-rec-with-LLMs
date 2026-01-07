"""
Generate per-book tags from aggregated review text (description field) using TF-IDF.

Usage:
    python scripts/generate_tags.py \
        --input data/books_processed.csv \
        --output data/books_processed.csv \
        --top-n 8

Notes:
- Uses unigrams + bigrams with English stopwords and a small domain stoplist.
- Filters out very short tokens and common boilerplate words.
- Writes a semicolon-joined `tags` column back to the CSV.
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_tags")

DOMAIN_STOPWORDS = {
    "book", "books", "story", "stories", "author", "authors", "novel", "fiction",
    "reader", "readers", "reading", "write", "writes", "writing", "written",
    "character", "characters", "plot", "series", "chapter", "chapters", "pages",
    "edition", "copy", "copies", "hardcover", "paperback", "kindle",
    # HTML / noise
    "amp", "nbsp", "lt", "gt",
    # Very common filler
    "com", "http", "https", "www",
}

TOKEN_RE = re.compile(r"^[a-zA-Z][a-zA-Z\-']{2,}$")


def normalize_text(text: str) -> str:
    """Clean text: HTML decode, strip control chars, collapse spaces."""
    t = html.unescape(str(text))
    t = unicodedata.normalize("NFKC", t)
    # Remove stray HTML entities and URLs
    t = re.sub(r"&[a-zA-Z]+;", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def filter_tokens(tokens: Iterable[str], max_tokens: int) -> List[str]:
    """Filter and deduplicate tokens, preserving order until max_tokens reached."""
    seen = set()
    result: List[str] = []
    for tok in tokens:
        t = tok.strip().lower()
        if not t:
            continue
        if t in seen:
            continue
        if t in DOMAIN_STOPWORDS:
            continue
        if len(t) < 3:
            continue
        if not TOKEN_RE.match(t):
            continue
        seen.add(t)
        result.append(t)
        if len(result) >= max_tokens:
            break
    return result


def compute_tags(corpus: List[str], top_n: int, max_features: int, min_df: int, max_df: float) -> List[str]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
        lowercase=True,
    )
    logger.info("Fitting TF-IDF on %d documents...", len(corpus))
    tfidf = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()

    tags: List[str] = []
    for i in range(tfidf.shape[0]):
        row = tfidf.getrow(i)
        if row.nnz == 0:
            tags.append("")
            continue
        data = row.data
        indices = row.indices
        # Pick top_n by weight
        if data.shape[0] <= top_n:
            top_local = np.argsort(data)[::-1]
        else:
            part = np.argpartition(data, -top_n)[-top_n:]
            top_local = part[np.argsort(data[part])[::-1]]
        ordered_tokens = [terms[indices[j]] for j in top_local]
        cleaned = filter_tokens(ordered_tokens, max_tokens=top_n)
        tags.append(";".join(cleaned))
    return tags


def main():
    parser = argparse.ArgumentParser(description="Generate per-book tags from descriptions")
    parser.add_argument("--input", type=Path, default=Path("data/books_processed.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/books_processed.csv"))
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.5)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    logger.info("Loading data from %s", args.input)
    df = pd.read_csv(args.input)
    if "description" not in df.columns:
        raise ValueError("Input CSV must have a 'description' column")

    corpus = [normalize_text(x) for x in df["description"].fillna("").astype(str).tolist()]
    tags = compute_tags(
        corpus,
        top_n=args.top_n,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    df["tags"] = tags
    logger.info("Writing tagged data to %s", args.output)
    df.to_csv(args.output, index=False)
    logger.info("Done. Sample tags: %s", tags[0:3])


if __name__ == "__main__":
    main()
