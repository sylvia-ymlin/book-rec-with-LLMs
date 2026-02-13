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


def run(
    input_path: Path = Path("data/books_processed.csv"),
    output_path: Path = Path("data/books_processed.csv"),
    batch_size: int = 16,
    device=None,
) -> None:
    """Generate emotion scores. Callable from Pipeline."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading data from %s", input_path)
    df = pd.read_csv(input_path)
    if "description" not in df.columns:
        raise ValueError("Input CSV must have a 'description' column")

    for col in TARGET_LABELS:
        if col not in df.columns:
            df[col] = 0.0

    model = load_model(device)
    texts = df["description"].fillna("").astype(str).tolist()
    n = len(df)

    logger.info("Scoring %d descriptions...", n)
    for start in tqdm(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        chunk = texts[start:end]
        outputs = model(chunk, truncation=True, max_length=512, top_k=None)
        for i, out in enumerate(outputs):
            vec = scores_to_vector(out)
            idx = start + i
            for col in TARGET_LABELS:
                df.at[idx, col] = vec[col]

    logger.info("Writing to %s", output_path)
    df.to_csv(output_path, index=False)


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
    dev = None
    if args.device:
        dev = "mps" if str(args.device).lower() == "mps" else (int(args.device) if str(args.device).isdigit() else None)
    run(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        device=dev,
    )


if __name__ == "__main__":
    main()
