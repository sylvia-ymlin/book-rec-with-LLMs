"""
Script to restore valid http image links from `data/books_basic_info.csv` into `data/books.db`.
This fixes the issue where processed data contained local file paths instead of original network URLs.
"""
import sqlite3
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.infra.config import DATA_DIR
from src.infra.utils import setup_logger

logger = setup_logger("fix_covers")

def fix_covers():
    db_path = DATA_DIR / "books.db"
    csv_path = DATA_DIR / "books_basic_info.csv"
    
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return
    
    if not csv_path.exists():
        logger.error(f"Source CSV not found at {csv_path}")
        return
        
    logger.info("Loading source CSV (this may take a moment)...")
    # Read strict columns to save memory
    try:
        df = pd.read_csv(csv_path, usecols=["isbn10", "isbn13", "image"], dtype=str)
    except ValueError:
        # Fallback if names differ
        df = pd.read_csv(csv_path, dtype=str)
        
    logger.info(f"Loaded {len(df)} rows from CSV.")
    
    # Filter for valid images only
    df = df[df["image"].str.startswith("http", na=False)]
    logger.info(f"Found {len(df)} rows with valid http links.")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    updated_count = 0
    
    # We'll prioritize ISBN13, then ISBN10
    # Prepare batch update
    updates = []
    
    logger.info("Preparing updates...")
    for _, row in df.iterrows():
        url = row["image"]
        isbn13 = row.get("isbn13")
        isbn10 = row.get("isbn10")
        
        if pd.notna(isbn13) and str(isbn13).strip():
            updates.append((url, str(isbn13)))
        elif pd.notna(isbn10) and str(isbn10).strip():
            # If we updates based on isbn10, we need a different query or handle it
            # Let's simple try to update by isbn10 if isbn13 matches nothing?
            # Actually, let's just create a list of (url, id, type)
            pass
            
    # Strategy: 
    # The DB looks like it might have 10-digit ISBNs in the 'isbn13' column for some rows.
    # So we use the valid ISBN from CSV (finding either 10 or 13) and match against BOTH columns in DB.
    
    updates = []
    
    for _, row in df.iterrows():
        url = row["image"]
        # Get whatever ID we have
        i13 = str(row.get("isbn13", "")).strip()
        i10 = str(row.get("isbn10", "")).strip()
        
        # Helper: is this a valid-looking ID?
        has_13 = len(i13) > 5 and i13.lower() != "nan"
        has_10 = len(i10) > 5 and i10.lower() != "nan"
        
        # We will try to match whatever we have against BOTH columns in DB to be safe
        ids_to_try = []
        if has_13: ids_to_try.append(i13)
        if has_10: ids_to_try.append(i10)
        
        # If we have both, we probably only need to queue one update set that tries both
        # But separate updates are safer if specific logic handles it
        # Actually, let's just create a flattened list of (url, id_to_match)
        # And run query: UPDATE books SET thumbnail = ? WHERE isbn13 = ? OR isbn10 = ?
        
        for pid in set(ids_to_try):
            updates.append((url, url, pid, pid))

    logger.info(f"Prepared {len(updates)} update params.")
    
    # Executing aggressive update
    # Note: We update `thumbnail` AND `image` columns.
    chunk_size = 10000
    for i in range(0, len(updates), chunk_size):
        chunk = updates[i:i + chunk_size]
        cursor.executemany("UPDATE books SET thumbnail = ?, image = ? WHERE isbn13 = ? OR isbn10 = ?", chunk)
        if i % 50000 == 0:
            logger.info(f"Processed {i} updates...")
            # conn.commit() # Optional commit/checkpoint

    conn.commit()
    logger.info("Update complete. Vacuuming...")
    
    # Check result
    cursor.execute("SELECT count(*) FROM books WHERE thumbnail LIKE 'http%'")
    count = cursor.fetchone()[0]
    logger.info(f"Total books with http thumbnails now: {count}")
    
    conn.close()

if __name__ == "__main__":
    fix_covers()
