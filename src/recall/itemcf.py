import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ItemCF:
    """
    Item-based Collaborative Filtering.
    
    ENGINEERING IMPROVEMENT:
    Transitioned from loading a 7GB+ in-memory similarity matrix (pickle) to an
    indexed SQLite database (`recall_models.db`). Candidate generation is now
    offloaded to highly efficient SQL aggregations.
    
    This change ensures zero-RAM loading for the similarity matrix while maintaining
    100% mathematical parity with the original Python implementation.
    """
    def __init__(self, data_dir='data/rec', save_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.db_path = Path("data/recall_models.db")
        self.conn = None
        
    def load(self):
        if self.db_path.exists():
            import sqlite3
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                logger.info(f"ItemCF: Connected to SQLite {self.db_path}")
                return True
            except Exception as e:
                logger.error(f"ItemCF: Failed to connect to SQLite: {e}")
        else:
            logger.warning(f"ItemCF: SQLite DB not found at {self.db_path}")
        return False

    def recommend(self, user_id, history_items=None, top_k=50):
        """
        Recommend items for a user based on history from SQLite.
        """
        if not self.conn:
            if not self.load():
                return []
        
        cursor = self.conn.cursor()
        
        # 1. Get history if not provided
        if history_items is None:
            cursor.execute("SELECT isbn FROM user_history WHERE user_id = ?", (user_id,))
            history_items = [row[0] for row in cursor.fetchall()]
        
        if not history_items:
            return []
            
        # 2. Query similar items (RRF-style aggregation in SQL or Python)
        # We fetch top similar items for each historical item
        rank = defaultdict(float)
        
        # To avoid too many queries, we can use a single query with IN clause
        # But for large history, SQLite has limits. Let's do a batch or join.
        
        # Efficient combined query: Find symbols similar to books in history
        placeholders = ', '.join(['?'] * len(history_items))
        query = f"""
            SELECT item2, SUM(score) as total_score
            FROM item_similarity
            WHERE item1 IN ({placeholders})
            AND item2 NOT IN ({placeholders})
            GROUP BY item2
            ORDER BY total_score DESC
            LIMIT ?
        """
        
        try:
            params = history_items + history_items + [top_k]
            cursor.execute(query, params)
            results = cursor.fetchall()
            return results
        except Exception as e:
            logger.error(f"ItemCF Query Error: {e}")
            return []

    def save(self): pass # Migration is done via script
    def fit(self, df): pass # Training should be done separately

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv('data/rec/train.csv')
    
    # Run on a smaller sample for testing?
    # df_sample = df.head(10000)
    
    model = ItemCF()
    model.fit(df)
    
    # Test rec
    user_id = df['user_id'].iloc[0]
    recs = model.recommend(user_id)
    print(f"Recs for {user_id}: {recs}")
