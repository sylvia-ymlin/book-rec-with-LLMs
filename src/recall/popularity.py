import pandas as pd
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PopularityRecall:
    def __init__(self, data_dir='data/rec', save_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.hot_items = []
        
    def fit(self, df):
        """
        Calculate popularity score
        df: pd.DataFrame with [isbn, rating, timestamp]
        """
        logger.info("Calculating popularity...")
        
        # Simple count * average rating approach? 
        # Or Just counts?
        # Let's use Count * Average Rating * Time Decay? 
        # Actually in recall, we just want "Hot items" to fill gaps. Counts are robust.
        
        # weighted_score = count
        stats = df.groupby('isbn').agg({
            'user_id': 'count',
            'rating': 'mean'
        }).rename(columns={'user_id': 'count', 'rating': 'avg_rating'})
        
        # Score = count * (avg_rating / 5) ?
        # Or just count. Let's use count.
        stats['score'] = stats['count']
        
        self.hot_items = stats.sort_values('score', ascending=False).index.tolist()
        
        self.save()
        return self.hot_items

    def recommend(self, user_id=None, top_k=50):
        # Return top K items
        # User ID is ignored (global popularity)
        return [(item, 1.0 / (i + 1)) for i, item in enumerate(self.hot_items[:top_k])]

    def save(self):
        with open(self.save_dir / 'popularity.pkl', 'wb') as f:
            pickle.dump(self.hot_items, f)
        logger.info(f"Model saved to {self.save_dir / 'popularity.pkl'}")

    def load(self):
        path = self.save_dir / 'popularity.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                self.hot_items = pickle.load(f)
            logger.info(f"Model loaded from {path}")
            return True
        return False
