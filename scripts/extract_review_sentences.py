"""
Extract representative review sentences from book descriptions using semantic similarity clustering.

Usage:
    python scripts/extract_review_sentences.py \
        --input data/books_processed.csv \
        --output data/books_processed.csv \
        --top-n 5 \
        --similarity-threshold 0.8

Notes:
- Splits descriptions into sentences
- Uses all-MiniLM-L6-v2 to vectorize sentences
- Clusters similar sentences (cosine similarity > threshold)
- Extracts representative sentences per book
- Stores as semicolon-separated review_highlights column
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("extract_review_sentences")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    if not text or pd.isna(text):
        return []
    
    text = str(text).strip()
    # Split on sentence boundaries (., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out very short sentences and clean
    sentences = [
        s.strip() 
        for s in sentences 
        if s.strip() and len(s.strip()) > 10
    ]
    return sentences


def cluster_sentences(sentences: List[str], embeddings: np.ndarray, threshold: float = 0.8) -> List[int]:
    """
    Cluster sentences by cosine similarity.
    Returns cluster ID for each sentence.
    """
    if len(sentences) == 0:
        return []
    if len(sentences) == 1:
        return [0]
    
    # Compute pairwise similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    # Simple clustering: assign each sentence to first similar cluster
    clusters = [-1] * len(sentences)
    current_cluster = 0
    
    for i in range(len(sentences)):
        if clusters[i] == -1:
            clusters[i] = current_cluster
            # Find all similar sentences
            for j in range(i + 1, len(sentences)):
                if clusters[j] == -1 and similarity_matrix[i, j] > threshold:
                    clusters[j] = current_cluster
            current_cluster += 1
    
    return clusters


def extract_representative_sentences(
    sentences: List[str], 
    embeddings: np.ndarray,
    clusters: List[int],
    top_n: int = 5
) -> List[str]:
    """
    Extract one representative sentence from each cluster,
    prioritizing longer/more informative sentences.
    """
    if not sentences:
        return []
    
    unique_clusters = set(clusters)
    representatives = []
    
    for cluster_id in sorted(unique_clusters):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        if not cluster_indices:
            continue
        
        # Pick longest sentence in cluster as representative
        best_idx = max(cluster_indices, key=lambda i: len(sentences[i]))
        representatives.append((best_idx, sentences[best_idx]))
    
    # Sort by original position and take top-n
    representatives.sort(key=lambda x: x[0])
    return [sent for _, sent in representatives[:top_n]]


def load_model(device: str | int | None):
    """Load sentence transformer model."""
    logger.info("Loading model: %s", MODEL_NAME)
    
    if isinstance(device, str) and device.lower() == "mps":
        logger.info("Using MPS (Apple GPU)")
        return SentenceTransformer(MODEL_NAME, device="mps")
    
    device_str = f"cuda:{device}" if isinstance(device, int) and device >= 0 else "cpu"
    logger.info(f"Using device: {device_str}")
    return SentenceTransformer(MODEL_NAME, device=device_str)


def main():
    parser = argparse.ArgumentParser(description="Extract representative review sentences")
    parser.add_argument("--input", type=Path, default=Path("data/books_processed.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/books_processed.csv"))
    parser.add_argument("--top-n", type=int, default=5, help="Top N sentences to extract per book")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, help="Cosine similarity threshold for clustering")
    parser.add_argument("--device", type=str, default="0", help="Device: 0 (CUDA), -1 (CPU), or mps (Apple)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit to N rows (for testing)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    logger.info("Loading data from %s", args.input)
    df = pd.read_csv(args.input)
    
    if args.max_rows:
        df = df.head(args.max_rows)
        logger.info(f"Limited to {args.max_rows} rows for testing")
    
    if "description" not in df.columns:
        raise ValueError("Input CSV must have a 'description' column")
    
    # Load model
    device = int(args.device) if args.device.lstrip('-').isdigit() else args.device
    model = load_model(device)
    
    # Process each book
    review_highlights = []
    
    logger.info(f"Processing {len(df)} books to extract review sentences...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        description = row["description"]
        
        # Split into sentences
        sentences = split_sentences(description)
        
        if not sentences:
            review_highlights.append("")
            continue
        
        # Embed sentences
        embeddings = model.encode(sentences, batch_size=args.batch_size, show_progress_bar=False)
        
        # Cluster similar sentences
        clusters = cluster_sentences(sentences, embeddings, threshold=args.similarity_threshold)
        
        # Extract representatives
        representatives = extract_representative_sentences(
            sentences, 
            embeddings, 
            clusters, 
            top_n=args.top_n
        )
        
        # Store as semicolon-separated string
        highlights_str = ";".join(representatives)
        review_highlights.append(highlights_str)
    
    # Add column to dataframe
    df["review_highlights"] = review_highlights
    
    logger.info("Writing output to %s", args.output)
    df.to_csv(args.output, index=False)
    
    # Print sample
    logger.info("Sample review highlights:")
    for i in range(min(3, len(df))):
        highlights = review_highlights[i]
        if highlights:
            print(f"\nBook {i+1}: {df.iloc[i]['title']}")
            for sent in highlights.split(";")[:2]:
                print(f"  • {sent[:80]}...")
    
    logger.info("Done. Added review_highlights column with %d entries", len(review_highlights))


if __name__ == "__main__":
    main()
