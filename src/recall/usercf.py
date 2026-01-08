import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UserCF:
    def __init__(self, data_dir='data/rec', save_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.u2u_sim = {}
        self.user_hist = {}
        
    def fit(self, df):
        """
        Calculate User-User Similarity Matrix
        """
        logger.info("Building UserCF similarity matrix...")
        
        # 1. Build Item -> Users inverted index
        item_users = defaultdict(list)
        self.user_hist = defaultdict(set)
        
        # Only use necessary columns
        # No time weight for UserCF simple version
        records = df[['user_id', 'isbn', 'rating']].values
        
        user_cnt = defaultdict(int)
        
        for user, item, rating in tqdm(records, desc="Building Inverted Index"):
            item_users[item].append(user)
            self.user_hist[user].add(item)
            user_cnt[user] += 1
            
        # 2. Calculate User Similarity
        sim = defaultdict(dict)
        
        # Optimization: Ignore items with too many users (Hot items don't help distiguish preference)
        # Threshold: e.g., > 1000 users?
        hot_item_limit = 200 
        
        num_items = len(item_users)
        logger.info(f"Total items: {num_items}")
        
        for item, users in tqdm(item_users.items(), desc="Calculating User Sim"):
            if len(users) > hot_item_limit:
                continue
            if len(users) < 2:
                continue
            
            # 1 / log(1 + n_users) - Penalize popular items
            weight = 1.0 / math.log(1 + len(users))
            
            # Iterating all pairs is O(N^2) where N is users per item
            # For 2000 users, 2M pairs per item. Might be slow if many such items.
            # Let's see.
            for i, u1 in enumerate(users):
                for j, u2 in enumerate(users):
                    if i == j: 
                        continue
                    
                    # Symmetric update
                    sim[u1][u2] = sim[u1].get(u2, 0) + weight
                    
        # 3. Normalize
        logger.info("Normalizing user similarity...")
        final_sim = {}
        for u1, related in tqdm(sim.items(), desc="Normalizing"):
            # Prune: Only keep top K similar users to save memory
            top_k_users = sorted(related.items(), key=lambda x: x[1], reverse=True)[:50]
            
            final_sim[u1] = {}
            for u2, score in top_k_users:
                norm = math.sqrt(user_cnt[u1] * user_cnt[u2])
                if norm > 0:
                    final_sim[u1][u2] = score / norm
        
        self.u2u_sim = final_sim
        self.save()
        return self.u2u_sim

    def recommend(self, user_id, history_items=None, top_k=50):
        """
        Recommend items based on similar users
        """
        rank = defaultdict(float)
        
        if user_id not in self.u2u_sim:
            return []
            
        similar_users = self.u2u_sim[user_id]
        
        # User history to exclude
        seen = self.user_hist.get(user_id, set())
        if history_items:
            seen.update(history_items)
            
        for sim_user, similarity in similar_users.items():
            # Get similar user's items
            sim_user_items = self.user_hist.get(sim_user, set())
            
            for item in sim_user_items:
                if item in seen:
                    continue
                
                # Weighted by user similarity
                # Could also weight by 'rating' of sim_user if available (but simpler to just count)
                rank[item] += similarity
        
        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save(self):
        with open(self.save_dir / 'usercf.pkl', 'wb') as f:
            pickle.dump({
                'u2u_sim': self.u2u_sim,
                'user_hist': self.user_hist
            }, f)
        logger.info(f"Model saved to {self.save_dir / 'usercf.pkl'}")

    def load(self):
        path = self.save_dir / 'usercf.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.u2u_sim = data['u2u_sim']
                self.user_hist = data['user_hist']
            logger.info(f"Model loaded from {path}")
            return True
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv('data/rec/train.csv')
    # Sampling for fast test?
    # df = df.head(50000)
    
    model = UserCF()
    model.fit(df)
    
    # Test rec
    if len(df) > 0:
        user_id = df['user_id'].iloc[0]
        recs = model.recommend(user_id)
        print(f"Recs for {user_id}: {recs}")
