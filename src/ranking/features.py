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
        self.has_sasrec = False # Initialize SASRec flag
        
    def _load_sasrec_features(self):
        """Load Pre-trained SASRec embeddings"""
        logger.info("Loading SASRec Embeddings...")
        try:
            import torch
            import pickle
            
            # Load user embeddings
            with open(self.data_dir / 'user_seq_emb.pkl', 'rb') as f:
                self.user_seq_emb = pickle.load(f)
                
            # Load Item embeddings from model checkpoint
            # We need the model state_dict to get 'item_emb.weight'
            # Note: item_emb.weight is [num_items+1, dim]
            # We also need item_map to map 'isbn' -> index
            
            with open(self.data_dir / 'item_map.pkl', 'rb') as f:
                self.sasrec_item_map = pickle.load(f)
                
            # Load Model State
            # Assuming sasrec_model.pth is in data/model/rec/sasrec_model.pth
            # self.model_dir is data/model/recall, so parent is data/model
            # Then we need to go to data/model/rec
            self.sasrec_model_path = self.model_dir.parent / 'rec' / 'sasrec_model.pth'
            state_dict = torch.load(self.sasrec_model_path, map_location='cpu')
            # Extract item_emb.weight
            self.sas_item_emb = state_dict['item_emb.weight'].numpy() # [N+1, H]
            
            self.has_sasrec = True
            logger.info("SASRec features loaded.")
            
        except Exception as e:
            logger.warning(f"Failed to load SASRec features: {e}")
            self.has_sasrec = False
    
    def _load_user_sequences(self):
        """Load user reading sequences (ordered by time) for Last-N similarity"""
        logger.info("Loading user sequences for Last-N similarity...")
        try:
            import pickle
            with open(self.data_dir / 'user_sequences.pkl', 'rb') as f:
                self.user_sequences = pickle.load(f)
            logger.info(f"Loaded sequences for {len(self.user_sequences)} users")
        except Exception as e:
            logger.warning(f"Failed to load user sequences: {e}")
            self.user_sequences = {}
    
    def _load_category_features(self):
        """Load book categories and user category preferences"""
        logger.info("Loading category features...")
        try:
            meta_df = pd.read_csv('data/books_processed.csv', usecols=['isbn13', 'simple_categories'])
            meta_df['isbn'] = meta_df['isbn13'].astype(str).str.replace(r'\.0$', '', regex=True)
            self.item_category = meta_df.set_index('isbn')['simple_categories'].fillna('Unknown').to_dict()
            
            # Build user category preferences from training data
            train_df = pd.read_csv(self.data_dir / 'train.csv')
            train_df['category'] = train_df['isbn'].map(self.item_category).fillna('Unknown')
            
            # User's preferred categories (set of categories they've rated)
            self.user_cat_prefs = train_df.groupby('user_id')['category'].apply(set).to_dict()
            
            logger.info(f"Loaded categories for {len(self.item_category)} items")
        except Exception as e:
            logger.warning(f"Failed to load category features: {e}")
            self.item_category = {}
            self.user_cat_prefs = {}

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
        
        # Load book metadata for content features
        self._load_content_features()
        
        # Load SASRec (NEW)
        self._load_sasrec_features()
        
        # Load user sequences for Last-N similarity (NEW)
        self._load_user_sequences()
        
        # Load category features (NEW)
        self._load_category_features()

    def _load_content_features(self):
        """Load book descriptions and other metadata"""
        logger.info("Loading content features (Complexity & Author)...")
        try:
            # Use processed data for descriptions and authors
            meta_df = pd.read_csv('data/books_processed.csv', usecols=['isbn13', 'description', 'authors'])
            # Clean ISBN
            meta_df['isbn'] = meta_df['isbn13'].astype(str).str.replace(r'\.0$', '', regex=True)
            
            # 1. Description Length
            meta_df['desc_len'] = meta_df['description'].fillna('').astype(str).apply(len)
            self.item_desc_len = meta_df.set_index('isbn')['desc_len'].to_dict()
            
            # 2. Author Map
            # Simplify: Take first author only to avoid complex split logic for now
            meta_df['author'] = meta_df['authors'].fillna('Unknown').str.split(',').str[0].str.strip()
            self.item_author = meta_df.set_index('isbn')['author'].to_dict()
            
            logger.info("Computing User-Author preferences...")
            # Pre-compute User-Author Ratings
            # We need train data
            train_df = pd.read_csv(self.data_dir / 'train.csv')
            train_df['author'] = train_df['isbn'].map(self.item_author).fillna('Unknown')
            
            # Group by [user, author] -> mean rating
            # This is a large dict, but manageable (100k users * avg 10 authors = 1M keys)
            self.user_author_stats = train_df.groupby(['user_id', 'author'])['rating'].mean().to_dict()
            
            # Pre-compute User Average Desc Length
            train_df['desc_len'] = train_df['isbn'].map(self.item_desc_len)
            user_aggs = train_df.groupby('user_id')['desc_len'].mean()
            self.user_avg_desc_len = user_aggs.to_dict()
            
            logger.info(f"Loaded content stats.")
            
        except Exception as e:
            logger.warning(f"Failed to load content features: {e}")
            self.item_desc_len = {}
            self.item_author = {}
            self.user_author_stats = {}
            self.user_avg_desc_len = {}

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
        
        # 3. Content Features: 
        # A. Complexity Match
        u_desc_len = self.user_avg_desc_len.get(user_id, 500)
        i_desc_len = self.item_desc_len.get(candidate_item, 0)
        feats['len_diff'] = abs(u_desc_len - i_desc_len)
        
        # B. Author Affinity (NEW)
        author = self.item_author.get(candidate_item, 'Unknown')
        # Look up user's rating for this author. Default to user's global mean.
        if (user_id, author) in self.user_author_stats:
            feats['u_auth_avg'] = self.user_author_stats[(user_id, author)]
            feats['u_auth_match'] = 1 # Indicator that use has rated this author
        else:
            feats['u_auth_avg'] = feats['u_mean'] # Fallback
            feats['u_auth_match'] = 0
            
        # 4. SASRec Similarity (NEW)
        if self.has_sasrec:
            # Get User Seq Embedding
            u_emb = self.user_seq_emb.get(user_id, None)
            
            # Get Item Embedding
            # Check map
            i_idx = self.sasrec_item_map.get(candidate_item, 0) # 0 is padding, usually items start from 1
            
            sas_score = 0.0
            if u_emb is not None and i_idx > 0:
                i_emb = self.sas_item_emb[i_idx]
                # Dot Product
                sas_score = float(np.dot(u_emb, i_emb))
                
            feats['sasrec_score'] = sas_score
        else:
            feats['sasrec_score'] = 0.0
        
        # 5. Last-N Similarity Features (NEW - from news rec)
        # Compute similarity between candidate and user's last N items
        sim_max, sim_min, sim_mean = 0.0, 0.0, 0.0
        if self.has_sasrec and hasattr(self, 'user_sequences'):
            user_seq = self.user_sequences.get(user_id, [])  # List of item indices
            i_idx = self.sasrec_item_map.get(candidate_item, 0)
            
            if len(user_seq) > 0 and i_idx > 0:
                cand_emb = self.sas_item_emb[i_idx]
                last_n_indices = user_seq[-5:]  # Last 5 item indices
                
                sims = []
                for hist_idx in last_n_indices:
                    # user_sequences already contains item indices, not ISBNs
                    if hist_idx > 0 and hist_idx < len(self.sas_item_emb):
                        hist_emb = self.sas_item_emb[hist_idx]
                        # Cosine similarity
                        norm_cand = np.linalg.norm(cand_emb)
                        norm_hist = np.linalg.norm(hist_emb)
                        if norm_cand > 0 and norm_hist > 0:
                            sim = float(np.dot(cand_emb, hist_emb) / (norm_cand * norm_hist))
                            sims.append(sim)
                
                if sims:
                    sim_max = max(sims)
                    sim_min = min(sims)
                    sim_mean = np.mean(sims)
        
        feats['sim_max'] = sim_max
        feats['sim_min'] = sim_min
        feats['sim_mean'] = sim_mean
        
        # 6. Category Affinity (NEW - from news rec)
        # Binary: is candidate's category in user's historical preferences
        is_cat_hob = 0
        if hasattr(self, 'item_category') and hasattr(self, 'user_cat_prefs'):
            cand_cat = self.item_category.get(candidate_item, 'Unknown')
            user_cats = self.user_cat_prefs.get(user_id, set())
            if cand_cat in user_cats:
                is_cat_hob = 1
        
        feats['is_cat_hob'] = is_cat_hob
        
        # 5. Interaction Features (ItemCF)
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
        
        # 6. Interaction Features (UserCF)
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
