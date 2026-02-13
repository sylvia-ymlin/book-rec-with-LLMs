#!/usr/bin/env python3
"""
Data Cleaning Script

Cleans and normalizes text data in the books dataset.
Handles: HTML entities, HTML tags, encoding issues, whitespace, special characters.

Usage:
    python data/scripts/clean_data.py [--input FILE] [--output FILE] [--dry-run]

This script should be run BEFORE other processing scripts.
"""

import argparse
import html
import re
import unicodedata
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("clean_data")


# =============================================================================
# Text Cleaning Functions
# =============================================================================

def clean_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    if not isinstance(text, str) or not text:
        return ""
    
    # Decode HTML entities (&amp; -> &, &nbsp; -> space, etc.)
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove common HTML artifacts
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # Remaining entities
    text = re.sub(r'&#\d+;', ' ', text)        # Numeric entities
    
    return text


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters."""
    if not isinstance(text, str):
        return ""
    
    # NFKC normalization (compatibility decomposition + canonical composition)
    # - Full-width -> half-width (Ａ -> A)
    # - Ligatures decomposed (ﬁ -> fi)
    # - Superscripts normalized (² -> 2)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove control characters (except newlines and tabs)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c in '\n\t')
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace characters."""
    if not isinstance(text, str):
        return ""
    
    # Replace various whitespace with regular space
    text = re.sub(r'[\t\r\f\v]+', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    if not isinstance(text, str):
        return ""
    
    # HTTP/HTTPS URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # www URLs
    text = re.sub(r'www\.\S+', '', text)
    
    return text


def fix_encoding_issues(text: str) -> str:
    """Fix common encoding issues (mojibake)."""
    if not isinstance(text, str):
        return ""
    
    # Common UTF-8 -> Latin-1 -> UTF-8 mojibake patterns
    replacements = {
        'â€™': "'",      # Right single quote
        'â€œ': '"',      # Left double quote
        'â€': '"',       # Right double quote
        'â€"': '—',      # Em dash
        'â€"': '–',      # En dash
        'â€¦': '...',    # Ellipsis
        'Ã©': 'é',       # e-acute
        'Ã¨': 'è',       # e-grave
        'Ã ': 'à',       # a-grave
        'Ã¢': 'â',       # a-circumflex
        'Ã®': 'î',       # i-circumflex
        'Ã´': 'ô',       # o-circumflex
        'Ã»': 'û',       # u-circumflex
        'Ã§': 'ç',       # c-cedilla
        'Ã±': 'ñ',       # n-tilde
        'â€˜': "'",      # Left single quote
        'Â ': ' ',       # Non-breaking space artifact
        'Â': '',         # Stray Â
    }
    
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    
    return text


def clean_text(text: str, 
               remove_html: bool = True,
               fix_encoding: bool = True,
               remove_url: bool = True,
               max_length: Optional[int] = None) -> str:
    """
    Apply all cleaning operations to text.
    
    Args:
        text: Input text
        remove_html: Remove HTML tags and decode entities
        fix_encoding: Fix common mojibake issues
        remove_url: Remove URLs
        max_length: Truncate to max length (None = no limit)
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Order matters!
    if fix_encoding:
        text = fix_encoding_issues(text)
    
    if remove_html:
        text = clean_html(text)
    
    if remove_url:
        text = remove_urls(text)
    
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + '...'
    
    return text


# =============================================================================
# Main Processing
# =============================================================================

def clean_dataframe(df: pd.DataFrame, 
                    text_columns: list,
                    max_lengths: Optional[dict] = None) -> pd.DataFrame:
    """
    Clean specified text columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        text_columns: List of column names to clean
        max_lengths: Optional dict of column -> max_length
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    max_lengths = max_lengths or {}
    
    for col in text_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue
        
        logger.info(f"Cleaning column: {col}")
        max_len = max_lengths.get(col)
        
        # Apply cleaning with progress bar
        tqdm.pandas(desc=f"  {col}")
        df[col] = df[col].progress_apply(lambda x: clean_text(x, max_length=max_len))
    
    return df


def analyze_data_quality(df: pd.DataFrame, text_columns: list) -> dict:
    """Analyze data quality before/after cleaning."""
    stats = {}
    
    for col in text_columns:
        if col not in df.columns:
            continue
        
        col_data = df[col].fillna('')
        stats[col] = {
            'total': len(col_data),
            'empty': (col_data == '').sum(),
            'avg_length': col_data.str.len().mean(),
            'has_html': col_data.str.contains(r'<[^>]+>', regex=True, na=False).sum(),
            'has_url': col_data.str.contains(r'https?://', regex=True, na=False).sum(),
        }
    
    return stats


def run(
    backup: bool = False,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    """Clean text data. Callable from Pipeline."""
    input_path = input_path or Path("data/books_processed.csv")
    output_path = output_path or input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} records")
    
    # Define columns to clean
    text_columns = ['title', 'description', 'authors', 'review_highlights']
    text_columns = [c for c in text_columns if c in df.columns]
    
    # Max lengths
    max_lengths = {
        'description': 5000,
        'review_highlights': 3000,
    }
    
    # Analyze before
    logger.info("\n📊 Data quality BEFORE cleaning:")
    stats_before = analyze_data_quality(df, text_columns)
    for col, s in stats_before.items():
        logger.info(f"  {col}: {s['has_html']} HTML, {s['has_url']} URLs, avg_len={s['avg_length']:.0f}")
    
    if dry_run:
        logger.info("\n[DRY RUN] No changes will be saved")
        return

    logger.info("\n🧹 Cleaning data...")
    df = clean_dataframe(df, text_columns, max_lengths)

    logger.info("\n📊 Data quality AFTER cleaning:")
    stats_after = analyze_data_quality(df, text_columns)
    for col, s in stats_after.items():
        logger.info(f"  {col}: {s['has_html']} HTML, {s['has_url']} URLs, avg_len={s['avg_length']:.0f}")

    if backup and output_path.exists():
        backup_path = output_path.with_suffix('.csv.bak')
        logger.info(f"Creating backup: {backup_path}")
        output_path.rename(backup_path)

    logger.info(f"\n💾 Saving to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("✅ Done!")


def main():
    parser = argparse.ArgumentParser(description="Clean text data in books dataset")
    parser.add_argument("--input", type=Path, default=Path("data/books_processed.csv"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Analyze without saving")
    parser.add_argument("--backup", action="store_true", help="Create backup before overwriting")
    args = parser.parse_args()
    run(
        backup=args.backup,
        input_path=args.input,
        output_path=args.output or args.input,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
