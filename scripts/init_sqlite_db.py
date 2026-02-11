import sqlite3
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_sqlite_db(data_dir: str = "data"):
    data_path = Path(data_dir)
    db_path = data_path / "books.db"
    processed_csv = data_path / "books_processed.csv"
    basic_info_csv = data_path / "books_basic_info.csv"

    if not processed_csv.exists():
        logger.error(f"Source file {processed_csv} not found.")
        return

    logger.info("Loading CSV data into memory for migration...")
    df_p = pd.read_csv(processed_csv)
    
    # Normalize ISBN13 in processed df
    df_p['isbn13'] = df_p['isbn13'].astype(str).str.strip().str.replace(".0", "", regex=False)

    if basic_info_csv.exists():
        logger.info(f"Merging with {basic_info_csv}...")
        df_b = pd.read_csv(basic_info_csv)
        df_b['isbn13'] = df_b['isbn13'].astype(str).str.strip().str.replace(".0", "", regex=False)
        # Keep only unique info from basic_info
        df_b = df_b[['isbn13', 'isbn10', 'image', 'publisher', 'publishedDate']].drop_duplicates(subset=['isbn13'])
        df = pd.merge(df_p, df_b, on='isbn13', how='left')
    else:
        df = df_p

    # Fill NaN
    df = df.fillna("")

    logger.info(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Create main books table
    logger.info("Creating 'books' table...")
    df.to_sql('books', conn, if_exists='replace', index=False)
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_isbn13 ON books (isbn13)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_isbn10 ON books (isbn10)")

    # 2. Create FTS5 virtual table for keyword search
    logger.info("Creating FTS5 virtual table 'books_fts'...")
    cursor.execute("DROP TABLE IF EXISTS books_fts")
    cursor.execute("""
        CREATE VIRTUAL TABLE books_fts USING fts5(
            isbn13 UNINDEXED,
            title,
            description,
            authors,
            simple_categories,
            content='books',
            tokenize='porter unicode61'
        )
    """)
    
    # Populate FTS index
    cursor.execute("""
        INSERT INTO books_fts(isbn13, title, description, authors, simple_categories)
        SELECT isbn13, title, description, authors, simple_categories FROM books
    """)

    conn.commit()
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM books")
    count = cursor.fetchone()[0]
    logger.info(f"Migration complete. {count} books imported into SQLite.")
    
    conn.close()

if __name__ == "__main__":
    init_sqlite_db()
