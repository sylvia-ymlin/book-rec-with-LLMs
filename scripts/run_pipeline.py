#!/usr/bin/env python3
"""
Unified Data Pipeline Runner

Executes the complete data processing pipeline in correct order.
Supports partial runs and validation between stages.

Usage:
    python scripts/run_pipeline.py                    # Full pipeline
    python scripts/run_pipeline.py --stage rec        # Only rec data
    python scripts/run_pipeline.py --skip-models      # Skip model training
    python scripts/run_pipeline.py --validate-only    # Only validate
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_script(script_path: str, description: str, args: list = None):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"   Script: {script_path}")
    print("=" * 60)
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description} (exit code: {result.returncode})")
        sys.exit(1)
    
    print(f"✅ Completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--stage", choices=[
        "all", "books", "rec", "index", "models"
    ], default="all", help="Which stage to run")
    parser.add_argument("--skip-models", action="store_true", help="Skip model training")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--device", default=None, help="Device for ML models (cpu/cuda/mps)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 DATA PIPELINE RUNNER")
    print("=" * 60)
    
    if args.validate_only:
        run_script("scripts/data/validate_data.py", "Validating all data")
        return
    
    start_total = time.time()
    
    # ==========================================================================
    # Stage 1: Book Data Processing
    # ==========================================================================
    if args.stage in ["all", "books"]:
        run_script(
            "scripts/data/clean_data.py",
            "Cleaning text data (HTML, encoding, whitespace)",
            args=["--backup"]
        )
        
        run_script(
            "scripts/data/build_books_basic_info.py",
            "Building books basic info"
        )
        
        device_args = ["--device", args.device] if args.device else []
        run_script(
            "scripts/data/generate_emotions.py",
            "Generating emotion scores",
            args=device_args
        )
        
        run_script(
            "scripts/data/generate_tags.py",
            "Generating tags"
        )
        
        run_script(
            "scripts/data/chunk_reviews.py",
            "Chunking reviews for Small-to-Big"
        )
    
    # ==========================================================================
    # Stage 2: RecSys Data Preparation
    # ==========================================================================
    if args.stage in ["all", "rec"]:
        run_script(
            "scripts/data/split_rec_data.py",
            "Splitting train/val/test data"
        )
        
        run_script(
            "scripts/data/build_sequences.py",
            "Building user sequences"
        )
    
    # ==========================================================================
    # Stage 3: Index Building
    # ==========================================================================
    if args.stage in ["all", "index"] and not args.skip_index:
        # Main book index is usually built by init_db.py or on startup
        # Here we build the chunk index for Small-to-Big
        run_script(
            "scripts/data/init_dual_index.py",
            "Building chunk vector index"
        )
    
    # ==========================================================================
    # Stage 4: Model Training
    # ==========================================================================
    if args.stage in ["all", "models"] and not args.skip_models:
        run_script(
            "scripts/model/build_recall_models.py",
            "Building ItemCF/UserCF models"
        )
        
        run_script(
            "scripts/model/train_youtube_dnn.py",
            "Training YoutubeDNN (requires GPU)"
        )
        
        run_script(
            "scripts/model/train_sasrec.py",
            "Training SASRec (requires GPU)"
        )
        
        run_script(
            "scripts/model/train_ranker.py",
            "Training LGBMRanker"
        )
    
    # ==========================================================================
    # Final Validation
    # ==========================================================================
    run_script(
        "scripts/data/validate_data.py",
        "Final validation"
    )
    
    elapsed_total = time.time() - start_total
    print("\n" + "=" * 60)
    print(f"🎉 PIPELINE COMPLETED in {elapsed_total/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
