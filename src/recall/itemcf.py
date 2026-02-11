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
    def __init__(self, data_dir='data/rec', save_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sim_matrix = {}
        self.user_hist = {}
        
    def fit(self, df):
        """
        Calculate Item-Item Similarity Matrix
        df: pd.DataFrame with columns [user_id, isbn, rating, timestamp]
        """
        logger.info("Building ItemCF similarity matrix...")
        
        # 1. Normalize timestamp to [0, 1] for weight calculation
        df = df.copy()
        df['timestamp'] = df['timestamp'].astype(float)
        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        df['norm_time'] = (df['timestamp'] - min_ts) / (max_ts - min_ts)
        
        # 2. Group by user to get interaction history
        # Sort by time for location weight
        df = df.sort_values(['user_id', 'timestamp'])
        
        user_grouped = df.groupby('user_id')
        self.user_hist = {} # Cache user history for prediction
        
        # sim[i][j]
        sim = defaultdict(dict)
        # item_cnt[i] for normalization
        item_cnt = defaultdict(int)
        
        for user_id, group in tqdm(user_grouped, desc="Calculating Co-occurrence"):
            # Get list of (isbn, rating, norm_time)
            records = group[['isbn', 'rating', 'norm_time']].values.tolist()
            self.user_hist[user_id] = set(r[0] for r in records)
            
            # Pairwise interaction
            for loc1, (item1, rating1, time1) in enumerate(records):
                item_cnt[item1] += 1
                
                for loc2, (item2, rating2, time2) in enumerate(records):
                    if item1 == item2:
                        continue
                    
                    # --- Weight Calculation (Transfer from News Rec) ---
                    
                    # 1. Location Weight: Closer position -> Higher weight
                    # loc_alpha = 1.0 if loc1 > loc2 else 0.7 (Directional? No, matrix should be symmetric usually, but news rec made it directional)
                    # Let's keep it symmetric for now or follow news rec exactly?
                    # News Rec: loc_alpha = 1.0 if loc1 > loc2 else 0.7  (Focus on "next prediction")
                    # But here we build a symmetric recommendation usually. 
                    # Let's use symmetric distance weight: 0.9 ** (|loc1 - loc2| - 1)
                    loc_weight = 0.9 ** (abs(loc1 - loc2) - 1)
                    
                    # 2. Time Weight: Closer time -> Higher weight
                    # Using exponential decay based on time diff
                    time_diff = abs(time1 - time2)
                    time_weight = np.exp(-time_diff) # Simple decay. 
                    # News rec used: np.exp(0.7 ** np.abs(time1 - time2)) which is weird. 
                    # Let's use 1 / (1 + time_diff * alpha) or exp(-alpha * diff)
                    # Since time is [0,1], diff is small. Let's stick to a simple form:
                    # 1 / (1 + 10 * time_diff) ensure it drops if diff is large
                    time_weight = 1 / (1 + 10 * time_diff)

                    # 3. Rating Weight (New): High ratings -> High weight
                    # rating is 1-5. 
                    rating_weight = (rating1 + rating2) / 10.0 # 0.2 to 1.0
                    
                    # 4. User Activity Penalty
                    user_weight = 1.0 / math.log(len(records) + 1)
                    
                    # Total Weight
                    weight = loc_weight * time_weight * rating_weight * user_weight
                    
                    sim[item1][item2] = sim[item1].get(item2, 0) + weight
        
        # 3. Normalize
        logger.info("Normalizing similarity matrix...")
        final_sim = {}
        for item1, related in tqdm(sim.items(), desc="Normalizing"):
            final_sim[item1] = {}
            for item2, score in related.items():
                # Cosine similarity normalization denominator
                # sqrt(count(i) * count(j))
                norm = math.sqrt(item_cnt[item1] * item_cnt[item2])
                if norm > 0:
                    final_sim[item1][item2] = score / norm
        
        self.sim_matrix = final_sim
        self.save()
        return self.sim_matrix

    def recommend(self, user_id, history_items=None, top_k=50):
        """
        Recommend items for a user based on history
        history_items: list of ISBNS given by user (if None, use stored training history)
        """
        rank = defaultdict(float)
        
        if history_items is None:
            # Check if we have history from training
            if user_id in self.user_hist:
                 # In training, history is stored as set
                 history_items = list(self.user_hist[user_id])
            else:
                return []
        
        # Iterate history items
        for item_i in history_items:
            # Find similar items
            if item_i in self.sim_matrix:
                for item_j, score in self.sim_matrix[item_i].items():
                    # Filter already seen
                    if item_j in history_items:
                        continue
                    rank[item_j] += score
        
        # Sort
        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save(self):
        with open(self.save_dir / 'itemcf.pkl', 'wb') as f:
            pickle.dump({
                'sim_matrix': self.sim_matrix,
                'user_hist': self.user_hist
            }, f)
        logger.info(f"Model saved to {self.save_dir / 'itemcf.pkl'}")

    def load(self):
        path = self.save_dir / 'itemcf.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.sim_matrix = data['sim_matrix']
                self.user_hist = data['user_hist']
            logger.info(f"Model loaded from {path}")
            return True
        return False

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
