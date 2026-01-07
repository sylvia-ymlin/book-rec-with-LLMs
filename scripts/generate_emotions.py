"""
Populate emotion scores (joy, sadness, fear, anger, surprise) from book descriptions.

Usage:
    python scripts/generate_emotions.py \
        --input data/books_processed.csv \
        --output data/books_processed.csv \
        --batch-size 16

Notes:
- Uses a lightweight transformer classifier (j-hartmann/emotion-english-distilroberta-base).
- Runs on CPU by default; set CUDA via env if available.
- Processes in batches to avoid memory spikes.
- Adds/overwrites columns: joy, sadness, fear, anger, surprise.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("generate_emotions")

TARGET_LABELS = ["joy", "sadness", "fear", "anger", "surprise"]
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


def load_model(device: str | int | None):
    logger.info("Loading model: %s", MODEL_NAME)

    if isinstance(device, str) and device.lower() == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available. Check PyTorch MPS build.")
        device_map = {"": "mps"}
        logger.info("Using MPS (Apple GPU)")
        return pipeline(
            "text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            return_all_scores=True,
            device_map=device_map,
            torch_dtype=torch.float16,
        )

    # CUDA or CPU path (device as int or None)
    device_id = device if isinstance(device, int) else -1
    if device_id >= 0:
        logger.info("Using CUDA device %s", device_id)
    else:
        logger.info("Using CPU")
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        return_all_scores=True,
        device=device_id,
    )


def scores_to_vector(scores: List[Dict[str, float]]) -> Dict[str, float]:
    # scores: list of dicts with keys label/score
    mapped = {k: 0.0 for k in TARGET_LABELS}
    for item in scores:
        label = item.get("label", "").lower()
        if label in mapped:
            mapped[label] = float(item.get("score", 0.0))
    return mapped


def main():
    ap = argparse.ArgumentParser(description="Generate emotion scores from descriptions")
    ap.add_argument("--input", type=Path, default=Path("data/books_processed.csv"))
    ap.add_argument("--output", type=Path, default=Path("data/books_processed.csv"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap for debugging")
    ap.add_argument("--device", default=None, help="'mps' for Apple GPU, CUDA device id, or omit for CPU")
    ap.add_argument("--checkpoint", type=int, default=5000, help="Rows between checkpoint writes")
    ap.add_argument("--resume", action="store_true", help="Resume if output exists (skip rows with scores)")
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    logger.info("Loading data from %s", args.input)
    df = pd.read_csv(args.input)
    if "description" not in df.columns:
        raise ValueError("Input CSV must have a 'description' column")

    if args.max_rows:
        df = df.head(args.max_rows)
        logger.info("Truncated to %d rows for max_rows", len(df))

    n = len(df)
    # Normalize device arg
    dev: str | int | None
    if args.device is None:
        dev = None
    else:
        if isinstance(args.device, str) and args.device.lower() == "mps":
            dev = "mps"
        else:
            try:
                dev = int(args.device)
            except ValueError:
                dev = None
    model = load_model(dev)

    # Prepare containers
    for col in TARGET_LABELS:
        if col not in df.columns:
            df[col] = 0.0

    # Resume support: if output exists, and resume flag set, load scores
    if args.resume and args.output.exists():
        logger.info("Resume enabled: loading existing output from %s", args.output)
        df_prev = pd.read_csv(args.output)
        for col in TARGET_LABELS:
            if col in df_prev.columns:
                df[col] = df_prev[col]

    texts = df["description"].fillna("").astype(str).tolist()
    batch = args.batch_size
    checkpoint = max(1, args.checkpoint)

    logger.info("Scoring %d descriptions (batch=%d, checkpoint=%d)...", n, batch, checkpoint)
    total_batches = (n + batch - 1) // batch
    for bidx, start in enumerate(tqdm(range(0, n, batch), total=total_batches)):
        end = min(start + batch, n)

        # Skip already-computed rows when resuming (all scores > 0)
        if args.resume:
            existing = df.loc[start:end-1, TARGET_LABELS].values
            if np.all(existing > 0):
                continue

        chunk = texts[start:end]
        outputs = model(chunk, truncation=True, max_length=512, top_k=None)
        for i, out in enumerate(outputs):
            vec = scores_to_vector(out)
            idx = start + i
            for col in TARGET_LABELS:
                df.at[idx, col] = vec[col]

        # periodic checkpoint write
        if (start > 0) and ((start % checkpoint) == 0):
            df.to_csv(args.output, index=False)

    logger.info("Writing to %s", args.output)
    df.to_csv(args.output, index=False)
    logger.info("Done. Example row: %s", df.head(1)[TARGET_LABELS].to_dict(orient="records"))


if __name__ == "__main__":
    main()
