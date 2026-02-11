#!/usr/bin/env python3
"""
Unified Data Pipeline Runner

Orchestrates Data Cleaning -> Training -> Evaluation using direct Python imports.
No subprocess calls. All logic invoked via Module.run() or src classes.

Usage:
    python scripts/run_pipeline.py                    # Full pipeline
    python scripts/run_pipeline.py --stage rec        # Only rec data
    python scripts/run_pipeline.py --skip-models      # Skip model training
    python scripts/run_pipeline.py --validate-only    # Only validate
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


class Pipeline:
    """
    Manages the full data pipeline: Data Cleaning -> Training -> Evaluation.

    All stages use direct Python imports; no subprocess.
    """

    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        device: str | None = None,
        skip_models: bool = False,
        skip_index: bool = False,
        stacking: bool = False,
        train_din: bool = False,
    ):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.rec_dir = self.data_dir / "rec"
        self.model_dir = self.data_dir / "model"
        self.device = device
        self.skip_models = skip_models
        self.skip_index = skip_index
        self.stacking = stacking
        self.train_din = train_din

    def _run_step(self, name: str, fn, *args, **kwargs):
        """Run a step with timing log."""
        logger.info("▶ %s", name)
        start = time.time()
        fn(*args, **kwargs)
        logger.info("  ✓ Done in %.1fs", time.time() - start)

    def run_data_cleaning(self, stage: str = "all") -> None:
        """Stage 1: Book data processing."""
        if stage not in ("all", "books"):
            return

        from scripts.data.clean_data import run as clean_run
        self._run_step("Clean text data", clean_run, backup=True)

        from scripts.data.build_books_basic_info import run as build_run
        raw_dir = self.data_dir / "raw"
        self._run_step("Build books basic info", build_run,
            books_path=raw_dir / "books_data.csv",
            ratings_path=raw_dir / "Books_rating.csv",
            output_path=self.data_dir / "books_basic_info.csv",
        )

        from scripts.data.generate_emotions import run as emotions_run
        self._run_step("Generate emotion scores", emotions_run, device=self.device)

        from scripts.data.generate_tags import run as tags_run
        self._run_step("Generate tags", tags_run)

        from scripts.data.chunk_reviews import chunk_reviews
        self._run_step("Chunk reviews", chunk_reviews,
            str(self.data_dir / "review_highlights.txt"),
            str(self.data_dir / "review_chunks.jsonl"),
        )

    def run_rec_preparation(self, stage: str = "all") -> None:
        """Stage 2: RecSys data preparation."""
        if stage not in ("all", "rec"):
            return

        from scripts.data.split_rec_data import run as split_run
        self._run_step("Split train/val/test", split_run,
            data_path=self.data_dir / "raw" / "Books_rating.csv",
            output_dir=self.rec_dir,
        )

        from scripts.data.build_sequences import build_sequences
        self._run_step("Build user sequences", build_sequences, str(self.rec_dir))

    def run_index_building(self, stage: str = "all") -> None:
        """Stage 3: Index building."""
        if stage not in ("all", "index") or self.skip_index:
            return

        from scripts.init_sqlite_db import init_sqlite_db
        self._run_step("Build SQLite metadata (books.db)", init_sqlite_db, str(self.data_dir))

        from scripts.data.init_dual_index import init_chunk_index
        self._run_step("Build chunk vector index", init_chunk_index)

    def run_training(self, stage: str = "all") -> None:
        """Stage 4: Model training via src imports."""
        if stage not in ("all", "models") or self.skip_models:
            return

        train_path = self.rec_dir / "train.csv"
        if not train_path.exists():
            logger.warning("train.csv not found, skipping model training")
            return

        import pandas as pd
        df = pd.read_csv(train_path)

        from src.recall.itemcf import ItemCF
        self._run_step("Train ItemCF", lambda: ItemCF().fit(df))

        from src.recall.usercf import UserCF
        self._run_step("Train UserCF", lambda: UserCF().fit(df))

        from src.recall.swing import Swing
        self._run_step("Train Swing", lambda: Swing().fit(df))

        from src.recall.popularity import PopularityRecall
        self._run_step("Train Popularity", lambda: PopularityRecall().fit(df))

        from src.recall.item2vec import Item2Vec
        self._run_step("Train Item2Vec", lambda: Item2Vec().fit(df))

        from src.recall.embedding import YoutubeDNNRecall
        self._run_step("Train YoutubeDNN", lambda: YoutubeDNNRecall().fit(
            df, books_path=self.data_dir / "books_processed.csv"
        ))

        from src.recall.sasrec_recall import SASRecRecall
        self._run_step("Train SASRec", lambda: SASRecRecall().fit(df))

        from scripts.model.train_ranker import train_ranker, train_stacking
        self._run_step("Train Ranker", train_stacking if self.stacking else train_ranker)

        from scripts.model.train_intent_router import main as train_intent
        self._run_step("Train intent classifier", train_intent)

        if getattr(self, "train_din", False):
            from scripts.model.train_din_ranker import train_din
            self._run_step("Train DIN ranker", lambda: train_din(
                data_dir=str(self.rec_dir),
                model_dir=str(self.model_dir),
                recall_dir=str(self.model_dir / "recall"),
            ))

    def run_evaluation(self) -> None:
        """Stage 5: Validation + RAG Golden Test Set (if exists)."""
        def _validate():
            from scripts.data.validate_data import (
                validate_raw, validate_processed, validate_rec,
                validate_index, validate_models,
            )
            validate_raw()
            validate_processed()
            validate_rec()
            validate_index()
            validate_models()

        self._run_step("Validate pipeline", _validate)

        # RAG Golden Test Set evaluation (optional)
        golden = self.rec_dir.parent / "rag_golden.csv"
        if not golden.exists():
            golden = self.rec_dir.parent / "rag_golden.example.csv"
        if golden.exists():
            def _run_rag_eval():
                from scripts.model.evaluate_rag import evaluate_rag
                m = evaluate_rag(str(golden))
                logger.info("RAG Accuracy@%d: %.4f  Recall@%d: %.4f  MRR@%d: %.4f",
                    m["top_k"], m["accuracy_at_k"], m["top_k"], m["recall_at_k"], m["top_k"], m["mrr_at_k"])
            self._run_step("RAG Golden Test Set", _run_rag_eval)

    def run(self, stage: str = "all") -> None:
        """Execute full pipeline: Data Cleaning -> Training -> Evaluation."""
        logger.info("=" * 60)
        logger.info("Pipeline: Data Cleaning -> Training -> Evaluation")
        logger.info("=" * 60)

        start_total = time.time()

        self.run_data_cleaning(stage)
        self.run_rec_preparation(stage)
        self.run_index_building(stage)
        self.run_training(stage)
        self.run_evaluation()

        elapsed = time.time() - start_total
        logger.info("=" * 60)
        logger.info("Pipeline completed in %.1f min", elapsed / 60)
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run data pipeline (no subprocess)")
    parser.add_argument(
        "--stage",
        choices=["all", "books", "rec", "index", "models"],
        default="all",
        help="Which stage to run",
    )
    parser.add_argument("--skip-models", action="store_true", help="Skip model training")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--device", default=None, help="Device for ML (cpu/cuda/mps)")
    parser.add_argument("--stacking", action="store_true", help="Enable stacking ranker")
    parser.add_argument("--din", action="store_true", help="Train DIN ranker (deep model)")
    args = parser.parse_args()

    if args.validate_only:
        logger.info("Validation only")
        Pipeline().run_evaluation()
        return

    pipeline = Pipeline(
        device=args.device,
        skip_models=args.skip_models,
        skip_index=args.skip_index,
        stacking=args.stacking,
        train_din=args.din,
    )
    pipeline.run(stage=args.stage)


if __name__ == "__main__":
    main()
