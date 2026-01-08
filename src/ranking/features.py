import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, data_dir='data/rec', model_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.recall_fusion = None
        
        # Stats cache
        self.user_stats = {}
        self.item_stats = {}
        
    def load_base_data(self):
        """Load train data to calculate static features"""
        logger.info("Loading base data for feature engineering...")
        df = pd.read_csv(self.data_dir / 'train.csv')
        
        # User Stats
        user_grp = df.groupby('user_id')
        self.user_stats = user_grp['rating'].agg(['count', 'mean', 'std']).to_dict('index')
        
        # Item Stats
        item_grp = df.groupby('isbn')
        self.item_stats = item_grp['rating'].agg(['count', 'mean', 'std']).to_dict('index')
        
        logger.info(f"Loaded stats for {len(self.user_stats)} users and {len(self.item_stats)} items")

        # Load recall models for interaction features
        from src.recall.fusion import RecallFusion
        self.recall_fusion = RecallFusion(self.data_dir, self.model_dir)
        self.recall_fusion.load_models()

    def generate_features(self, user_id, candidate_item):
        """
        Generate feature vector for a (user, item) pair
        Returns: dict of features
        """
        feats = {}
        
        # 1. User Features
        u_stat = self.user_stats.get(user_id, {'count': 0, 'mean': 0, 'std': 0})
        feats['u_cnt'] = np.log1p(u_stat['count'])
        feats['u_mean'] = u_stat['mean']
        feats['u_std'] = u_stat['std'] if not pd.isna(u_stat['std']) else 0
        
        # 2. Item Features
        i_stat = self.item_stats.get(candidate_item, {'count': 0, 'mean': 0, 'std': 0})
        feats['i_cnt'] = np.log1p(i_stat['count'])
        feats['i_mean'] = i_stat['mean']
        feats['i_std'] = i_stat['std'] if not pd.isna(i_stat['std']) else 0
        
        # 3. Interaction Features (ItemCF)
        # Calculate similarity between candidate and user history
        # We can reuse the stored matrix in ItemCF
        # sum(sim(hist_item, candidate))
        icf_score = 0
        icf_max = 0
        
        # Access ItemCF internal matrix directly for speed
        itemcf = self.recall_fusion.itemcf
        history = itemcf.user_hist.get(user_id, set())
        
        if candidate_item in itemcf.sim_matrix:
            related = itemcf.sim_matrix[candidate_item]
            # Since matrix is symmetric, sim[cand][hist] should act as sim[hist][cand]
            # Wait, my implementation stored sim[i][j]. Is it symmetric? 
            # In fit(), I did: sim[item1][item2] += ...
            # I iterated all pairs. So if I iterated (A, B) and (B, A), it is symmetric?
            # Yes, nested loop: for loc1... for loc2...
            # And weights are symmetric (abs diff).
            # So sim[A][B] == sim[B][A].
            
            sims = []
            for hist_item in history:
                if hist_item in related:
                    sims.append(related[hist_item])
            
            if sims:
                icf_score = sum(sims)
                icf_max = max(sims)
                
        feats['icf_sum'] = icf_score
        feats['icf_max'] = icf_max
        
        # 4. Interaction Features (UserCF)
        # Similar users who rated this item
        ucf_score = 0
        usercf = self.recall_fusion.usercf
        
        # Inverted index: candidate -> users who rated it
        # I didn't store inverted index in UserCF model save?
        # Check UserCF.save(): {'u2u_sim': ..., 'user_hist': ...}
        # Inverted index is not saved. 
        # But UserCF uses u2u_sim[user] -> similar_users.
        # Then check if similar_users rated candidate.
        
        if user_id in usercf.u2u_sim:
            sim_users = usercf.u2u_sim[user_id]
            for sim_u, sim_score in sim_users.items():
                if candidate_item in usercf.user_hist.get(sim_u, set()):
                    ucf_score += sim_score
                    
        feats['ucf_sum'] = ucf_score
        
        return feats

    def create_dateset(self, df_samples):
        """
        Create DataFrame with features
        df_samples: pd.DataFrame with [user_id, isbn, label]
        """
        if not self.user_stats:
            self.load_base_data()
            
        data = []
        labels = []
        
        # Batch processing?
        # For simple purpose, loop is fine
        for _, row in tqdm(df_samples.iterrows(), total=len(df_samples), desc="Generating Features"):
            user = row['user_id']
            item = row['isbn']
            label = row.get('label', 0)
            
            f = self.generate_features(user, item)
            f['label'] = label
            data.append(f)
            
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    fe = FeatureEngineer()
    
    # Create dummy samples
    samples = pd.DataFrame({
        'user_id': ['A1ZQ1LUQ9R6JHZ'], 
        'isbn': ['0001047604'],
        'label': [1]
    })
    
    df_feats = fe.create_dateset(samples)
    print(df_feats.head())
