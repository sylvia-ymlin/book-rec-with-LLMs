#!/usr/bin/env python3
"""
Data Pipeline Validation Script

Validates data quality at each stage of the pipeline.
Run after data processing to ensure correctness.

Usage:
    python scripts/data/validate_data.py [--stage all|raw|processed|rec|index]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.data_config import *


class ValidationError(Exception):
    pass


def check(condition: bool, message: str, warn_only: bool = False):
    """Assert with custom message."""
    if not condition:
        if warn_only:
            print(f"  ⚠️  WARNING: {message}")
        else:
            raise ValidationError(f"❌ FAILED: {message}")
    return condition


def validate_raw():
    """Validate raw input data exists and has expected schema."""
    print("\n📁 Validating RAW data...")
    
    # Check files exist
    check(RAW_BOOKS.exists(), f"Raw books file not found: {RAW_BOOKS}")
    check(RAW_RATINGS.exists(), f"Raw ratings file not found: {RAW_RATINGS}")
    
    # Check books schema
    books = pd.read_csv(RAW_BOOKS, nrows=100, engine="python", on_bad_lines="skip")
    required_cols = ["Title", "description", "authors", "categories"]
    for col in required_cols:
        check(col in books.columns, f"Missing column in books_data.csv: {col}")
    
    # Check ratings schema
    ratings = pd.read_csv(RAW_RATINGS, nrows=100, engine="python", on_bad_lines="skip")
    required_cols = ["Id", "User_id", "review/score", "review/time"]
    for col in required_cols:
        check(col in ratings.columns, f"Missing column in Books_rating.csv: {col}")
    
    print("  ✅ Raw data validation passed")
    return True


def validate_processed():
    """Validate processed book data."""
    print("\n📚 Validating PROCESSED data...")
    
    check(BOOKS_PROCESSED.exists(), f"Processed books not found: {BOOKS_PROCESSED}")
    
    df = pd.read_csv(BOOKS_PROCESSED)
    n = len(df)
    print(f"  Total books: {n:,}")
    
    # Check required columns
    required = ["title", "description", "authors"]
    for col in required:
        check(col in df.columns, f"Missing column: {col}")
    
    # Categories may be named differently
    has_categories = "categories" in df.columns or "simple_categories" in df.columns
    check(has_categories, "Missing categories column", warn_only=True)
    
    # Check data quality
    desc_fill_rate = df["description"].notna().mean()
    check(desc_fill_rate > 0.7, f"Description fill rate too low: {desc_fill_rate:.1%}", warn_only=True)
    print(f"  Description fill rate: {desc_fill_rate:.1%}")
    
    # Check emotion columns if exist
    if all(col in df.columns for col in EMOTION_LABELS):
        emotion_fill = df[EMOTION_LABELS].notna().all(axis=1).mean()
        print(f"  Emotion fill rate: {emotion_fill:.1%}")
        check(emotion_fill > 0.5, f"Emotion fill rate too low: {emotion_fill:.1%}", warn_only=True)
    else:
        print("  ⚠️  Emotion columns not found (run generate_emotions.py)")
    
    # Check tags column
    if "tags" in df.columns:
        tags_fill = (df["tags"].notna() & (df["tags"] != "")).mean()
        print(f"  Tags fill rate: {tags_fill:.1%}")
    else:
        print("  ⚠️  Tags column not found (run generate_tags.py)")
    
    print("  ✅ Processed data validation passed")
    return True


def validate_rec():
    """Validate recommendation system data splits."""
    print("\n🎯 Validating REC data...")
    
    check(REC_DIR.exists(), f"Rec directory not found: {REC_DIR}")
    check(TRAIN_CSV.exists(), f"Train data not found: {TRAIN_CSV}")
    check(VAL_CSV.exists(), f"Validation data not found: {VAL_CSV}")
    check(TEST_CSV.exists(), f"Test data not found: {TEST_CSV}")
    
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)
    
    print(f"  Train: {len(train):,} records")
    print(f"  Val: {len(val):,} records")
    print(f"  Test: {len(test):,} records")
    
    # Check no data leakage: same user should not appear in train and test with same item
    train_pairs = set(zip(train["user_id"], train["isbn"]))
    test_pairs = set(zip(test["user_id"], test["isbn"]))
    overlap = train_pairs & test_pairs
    check(len(overlap) == 0, f"Data leakage detected! {len(overlap)} overlapping user-item pairs")
    
    # Check user counts
    train_users = set(train["user_id"])
    val_users = set(val["user_id"])
    test_users = set(test["user_id"])
    
    check(val_users == test_users, "Val and test users should be the same")
    check(train_users >= test_users, "All test users should be in train")
    
    print(f"  Unique users: {len(test_users):,}")
    print(f"  Unique items: {train['isbn'].nunique():,}")
    
    # Check sequences if exist
    if USER_SEQUENCES.exists():
        import pickle
        with open(USER_SEQUENCES, "rb") as f:
            seqs = pickle.load(f)
        print(f"  User sequences: {len(seqs):,}")
        avg_len = np.mean([len(s) for s in seqs.values()])
        print(f"  Avg sequence length: {avg_len:.1f}")
    else:
        print("  ⚠️  User sequences not found (run build_sequences.py)")
    
    print("  ✅ Rec data validation passed")
    return True


def validate_index():
    """Validate vector indices."""
    print("\n🔍 Validating INDEX data...")
    
    # Check ChromaDB
    if CHROMA_DB.exists():
        # Count files in chroma_db
        db_files = list(CHROMA_DB.glob("**/*"))
        print(f"  ChromaDB files: {len(db_files)}")
        check(len(db_files) > 0, "ChromaDB appears empty")
    else:
        print("  ⚠️  ChromaDB not found (run init_db.py or init_dual_index.py)")
    
    # Check chunks index
    if CHROMA_CHUNKS.exists():
        chunk_files = list(CHROMA_CHUNKS.glob("**/*"))
        print(f"  Chunk index files: {len(chunk_files)}")
    else:
        print("  ⚠️  Chunk index not found (run init_dual_index.py)")
    
    # Check review chunks source
    if REVIEW_CHUNKS.exists():
        with open(REVIEW_CHUNKS, "r") as f:
            chunk_count = sum(1 for _ in f)
        print(f"  Review chunks: {chunk_count:,}")
    else:
        print("  ⚠️  Review chunks JSONL not found (run chunk_reviews.py)")
    
    print("  ✅ Index validation passed")
    return True


def validate_models():
    """Validate trained models exist."""
    print("\n🤖 Validating MODELS...")
    
    models = [
        ("ItemCF", ITEMCF_MODEL),
        ("UserCF", USERCF_MODEL),
        ("YoutubeDNN", YOUTUBE_DNN_MODEL),
        ("SASRec", SASREC_MODEL),
        ("XGBoost", XGB_RANKER),
    ]
    
    for name, path in models:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ⚠️  {name}: NOT FOUND")
    
    print("  ✅ Model check completed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate data pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "raw", "processed", "rec", "index", "models"],
        default="all",
        help="Which stage to validate"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 DATA PIPELINE VALIDATION")
    print("=" * 60)
    
    try:
        if args.stage in ["all", "raw"]:
            validate_raw()
        if args.stage in ["all", "processed"]:
            validate_processed()
        if args.stage in ["all", "rec"]:
            validate_rec()
        if args.stage in ["all", "index"]:
            validate_index()
        if args.stage in ["all", "models"]:
            validate_models()
        
        print("\n" + "=" * 60)
        print("✅ ALL VALIDATIONS PASSED")
        print("=" * 60)
        
    except ValidationError as e:
        print(f"\n{e}")
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
