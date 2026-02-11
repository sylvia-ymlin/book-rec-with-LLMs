"""
Shared utilities for scripts/. Reduces duplication across data/model scripts.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root on path for config imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def get_project_root() -> Path:
    """Project root directory."""
    return _PROJECT_ROOT


def get_data_dir() -> Path:
    """Data directory (data/)."""
    return _PROJECT_ROOT / "data"


def setup_script_logger(
    name: str,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt: str = "%H:%M:%S",
) -> logging.Logger:
    """
    Configure logging for a script. Use instead of ad-hoc logging.basicConfig.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_str, datefmt=datefmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_data_config():
    """Lazy-load config.data_config paths. Use when script needs DATA_DIR, BOOKS_PROCESSED, etc."""
    from config.data_config import (
        DATA_DIR,
        RAW_DIR,
        BOOKS_PROCESSED,
        BOOKS_BASIC_INFO,
        REC_DIR,
        RAW_BOOKS,
        RAW_RATINGS,
    )
    return {
        "data_dir": DATA_DIR,
        "raw_dir": RAW_DIR,
        "books_processed": BOOKS_PROCESSED,
        "books_basic_info": BOOKS_BASIC_INFO,
        "rec_dir": REC_DIR,
        "raw_books": RAW_BOOKS,
        "raw_ratings": RAW_RATINGS,
    }
