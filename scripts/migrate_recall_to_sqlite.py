import pickle
import sqlite3
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_itemcf(pkl_path, db_path):
    pkl_path = Path(pkl_path)
    db_path = Path(db_path)
    
    if not pkl_path.exists():
        logger.error(f"Pickle not found at {pkl_path}")
        return

    logger.info(f"Loading massive pickle {pkl_path} (1.4GB)... This may take a minute.")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    sim_matrix = data.get('sim_matrix', {})
    user_hist = data.get('user_hist', {})
    
    logger.info(f"Connecting to SQLite {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("DROP TABLE IF EXISTS item_similarity")
    cursor.execute("""
        CREATE TABLE item_similarity (
            item1 TEXT,
            item2 TEXT,
            score REAL
        )
    """)
    
    cursor.execute("DROP TABLE IF EXISTS user_history")
    cursor.execute("""
        CREATE TABLE user_history (
            user_id TEXT,
            isbn TEXT
        )
    """)
    
    # Insert Item Similarity
    logger.info("Inserting item similarity data...")
    batch = []
    for item1, related in tqdm(sim_matrix.items(), desc="ItemCF Similarity"):
        for item2, score in related.items():
            batch.append((item1, item2, score))
            if len(batch) >= 100000:
                cursor.executemany("INSERT INTO item_similarity VALUES (?, ?, ?)", batch)
                batch = []
    if batch:
        cursor.executemany("INSERT INTO item_similarity VALUES (?, ?, ?)", batch)
    
    # Insert User History
    logger.info("Inserting user history data...")
    batch = []
    for user_id, isbns in tqdm(user_hist.items(), desc="User History"):
        for isbn in isbns:
            batch.append((user_id, isbn))
            if len(batch) >= 100000:
                cursor.executemany("INSERT INTO user_history VALUES (?, ?)", batch)
                batch = []
    if batch:
        cursor.executemany("INSERT INTO user_history VALUES (?, ?)", batch)
    
    # Create Indices
    logger.info("Creating indices...")
    cursor.execute("CREATE INDEX idx_item1 ON item_similarity(item1)")
    cursor.execute("CREATE INDEX idx_user ON user_history(user_id)")
    
    conn.commit()
    conn.close()
    logger.info("Migration complete.")

if __name__ == "__main__":
    migrate_itemcf("data/model/recall/itemcf.pkl", "data/recall_models.db")
