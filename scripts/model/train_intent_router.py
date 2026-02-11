#!/usr/bin/env python3
"""
Train model-based intent classifier for Query Router.

Replaces rule-based heuristics with TF-IDF + LogisticRegression (or FastText/DistilBERT).
Uses synthetic seed data; extend with real labeled queries via --data CSV.

Usage:
    python scripts/model/train_intent_router.py
    python scripts/model/train_intent_router.py --data data/intent_labels.csv
    python scripts/model/train_intent_router.py --backend fasttext
    python scripts/model/train_intent_router.py --backend distilbert

Output:
    data/model/intent_classifier.pkl  (or .bin for fasttext)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import logging

import pandas as pd

from src.core.intent_classifier import train_classifier, INTENTS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Synthetic training data: (query, intent)
# Extend with real user queries for better generalization
SEED_DATA = [
    # small_to_big: detail-oriented, plot/review focused
    ("book with twist ending", "small_to_big"),
    ("unreliable narrator", "small_to_big"),
    ("spoiler about the ending", "small_to_big"),
    ("what did readers think", "small_to_big"),
    ("opinion on the book", "small_to_big"),
    ("hidden details in the story", "small_to_big"),
    ("did anyone cry reading this", "small_to_big"),
    ("review of the book", "small_to_big"),
    ("plot twist reveal", "small_to_big"),
    ("unreliable narrator twist", "small_to_big"),
    ("readers who loved the ending", "small_to_big"),
    ("spoiler what happens at the end", "small_to_big"),
    # fast: short keyword queries
    ("AI book", "fast"),
    ("Python", "fast"),
    ("romance", "fast"),
    ("machine learning", "fast"),
    ("science fiction", "fast"),
    ("best AI book", "fast"),
    ("Python programming", "fast"),
    ("self help", "fast"),
    ("business", "fast"),
    ("fiction", "fast"),
    ("thriller", "fast"),
    ("mystery novel", "fast"),
    ("finance", "fast"),
    ("history", "fast"),
    ("psychology", "fast"),
    ("data science", "fast"),
    ("cooking", "fast"),
    ("music", "fast"),
    ("art", "fast"),
    ("philosophy", "fast"),
    # deep: natural language, complex queries
    ("What are the best books about artificial intelligence for beginners", "deep"),
    ("I'm looking for something similar to Harry Potter", "deep"),
    ("Books that help you understand machine learning", "deep"),
    ("Recommend me a book like Sapiens but about technology", "deep"),
    ("I want to learn about psychology and human behavior", "deep"),
    ("What should I read if I liked 1984", "deep"),
    ("Looking for books on startup founding and entrepreneurship", "deep"),
    ("Can you suggest books about climate change and sustainability", "deep"),
    ("I need a book that explains quantum physics simply", "deep"),
    ("Books for someone who wants to improve their writing skills", "deep"),
    ("What are some good fiction books set in Japan", "deep"),
    ("Recommendations for someone getting into philosophy", "deep"),
    ("Books that discuss the future of work and automation", "deep"),
    ("I'm interested in biographies of scientists", "deep"),
    ("Something light and funny for a long flight", "deep"),
    ("Books about the history of mathematics", "deep"),
    ("Recommend me novels with strong female protagonists", "deep"),
    ("What to read to understand economics", "deep"),
    ("Books on meditation and mindfulness", "deep"),
]


def load_training_data(data_path: Path | None) -> tuple[list[str], list[str]]:
    """Load (queries, labels) from SEED_DATA + optional CSV."""
    queries = [q for q, _ in SEED_DATA]
    labels = [l for _, l in SEED_DATA]

    if data_path and data_path.exists():
        df = pd.read_csv(data_path)
        q_col = "query" if "query" in df.columns else df.columns[0]
        l_col = "intent" if "intent" in df.columns else df.columns[1]
        extra_q = df[q_col].astype(str).tolist()
        extra_l = df[l_col].astype(str).tolist()
        queries.extend(extra_q)
        labels.extend(extra_l)
        logger.info("Loaded %d extra samples from %s", len(extra_q), data_path)

    return queries, labels


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--data", type=Path, default=None, help="CSV with query,intent columns")
    parser.add_argument("--backend", choices=["tfidf", "fasttext", "distilbert"], default="tfidf")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / "data" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries, labels = load_training_data(args.data)

    logger.info("Training intent classifier (%s) on %d samples...", args.backend, len(queries))
    result = train_classifier(queries, labels, backend=args.backend)

    if args.backend == "fasttext":
        out_path = out_dir / "intent_classifier.bin"
        result.save_model(str(out_path))
    else:
        out_path = out_dir / "intent_classifier.pkl"
        if args.backend == "distilbert":
            joblib.dump(result, out_path)  # dict with pipeline, backend, etc.
        else:
            joblib.dump({"pipeline": result, "backend": "tfidf"}, out_path)

    logger.info("Saved to %s", out_path)

    # Quick sanity check
    for intent in INTENTS:
        sample = next((q for q, l in zip(queries, labels) if l == intent), None)
        if sample:
            if args.backend == "fasttext":
                pred = result.predict(sample)[0][0].replace("__label__", "")
            elif args.backend == "distilbert":
                from transformers import pipeline
            pipe = pipeline("zero-shot-classification", model="distilbert-base-uncased", device=-1)
            pred = pipe(sample, INTENTS, multi_label=False)["labels"][0]
            else:
                pred = result.predict([sample])[0]
            ok = "✓" if pred == intent else "✗"
            logger.info("  %s %s: %r -> %s", ok, intent, sample[:40], pred)


if __name__ == "__main__":
    main()
