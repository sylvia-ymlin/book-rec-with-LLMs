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
    


    def load_base_data(self):
        """Load feature maps via MetadataStore singleton"""
        logger.info("Accessing MetadataStore for ranking features...")
        from src.core.metadata_store import metadata_store
        
        # Bind References directly from the global singleton store
        self.user_stats = metadata_store.user_stats
        self.item_stats = metadata_store.item_stats
        self.item_category = metadata_store.item_category
        self.user_cat_prefs = {} # Fallback
        self.item_desc_len = {}  # Fallback
        self.item_author = metadata_store.item_author
        self.user_author_stats = {}
        self.user_avg_desc_len = {}
        
        logger.info(f"FeatureEngineer: Linked to MetadataStore maps.")

        # Load recall models for interaction features
        from src.recall.fusion import RecallFusion
        self.recall_fusion = RecallFusion(self.data_dir, self.model_dir)
        self.recall_fusion.load_models()
        
        # Load SASRec
        self._load_sasrec_features()
        
        # Load user sequences
        self._load_user_sequences()



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
        usercf = self.recall_fusion.usercf
        
        # FIX: ItemCF in Zero-RAM mode (SQLite) doesn't have user_hist or sim_matrix.
        # Use UserCF's user_hist as fallback or empty set.
        history = set()
        if hasattr(usercf, "user_hist"):
             history = usercf.user_hist.get(user_id, set())
        
        # Check if ItemCF has in-memory matrix (Legacy)
        if hasattr(itemcf, "sim_matrix") and candidate_item in itemcf.sim_matrix:
            related = itemcf.sim_matrix[candidate_item]
            # Since matrix is symmetric, sim[cand][hist] should act as sim[hist][cand]
            
            sims = []
            for hist_item in history:
                if hist_item in related:
                    sims.append(related[hist_item])
            
            if sims:
                icf_score = sum(sims)
                icf_max = max(sims)
        else:
             # Zero-RAM mode: Can't easily compute ICF features per-item without SQL overhead.
             # Skipping ICF features for ranking to prioritize stability.
             pass
                
        feats['icf_sum'] = icf_score
        feats['icf_max'] = icf_max
        
        # 6. Interaction Features (UserCF)
        # Similar users who rated this item
        ucf_score = 0
        # usercf already defined above
        
        if hasattr(usercf, "u2u_sim") and user_id in usercf.u2u_sim:
            sim_users = usercf.u2u_sim[user_id]
            for sim_u, sim_score in sim_users.items():
                if hasattr(usercf, "user_hist") and candidate_item in usercf.user_hist.get(sim_u, set()):
                    ucf_score += sim_score
                    
        feats['ucf_sum'] = ucf_score
        
        return feats

    def generate_features_batch(self, user_id, candidate_items):
        """
        Optimized batch feature generation for a single user and multiple items.
        Significantly faster than calling generate_features in a loop.
        """
        import numpy as np
        
        # 1. Pre-fetch User Features (O(1))
        u_stat = self.user_stats.get(user_id, {'count': 0, 'mean': 0, 'std': 0})
        u_cnt = np.log1p(u_stat['count'])
        u_mean = u_stat['mean']
        u_std = u_stat['std'] if not pd.isna(u_stat['std']) else 0
        
        u_desc_len = self.user_avg_desc_len.get(user_id, 500)
        user_cats = self.user_cat_prefs.get(user_id, set())
        
        # 2. Pre-fetch Interaction Data
        # itemcf deleted here as not used for history
        usercf = self.recall_fusion.usercf
        
        # FIX: Use UserCF history (ItemCF is zero-RAM)
        history = set()
        if hasattr(usercf, "user_hist"):
            history = usercf.user_hist.get(user_id, set())
        
        usercf_sim_users = {}
        if hasattr(usercf, "u2u_sim") and user_id in usercf.u2u_sim:
            usercf_sim_users = usercf.u2u_sim[user_id] 
            # Pre-filter? No, we iterate candidates.
            
        # 3. Batch SASRec (Vectorized)
        sasrec_scores = np.zeros(len(candidate_items))
        has_sas = False
        if self.has_sasrec:
            u_emb = self.user_seq_emb.get(user_id, None)
            if u_emb is not None:
                # Get valid indices
                indices = [self.sasrec_item_map.get(item, 0) for item in candidate_items]
                valid_mask = [i > 0 for i in indices]
                
                if any(valid_mask):
                    idx_array = np.array(indices)
                    # Filter 0 indices for safety (though 0 is usually padding/unknown)
                    # Actually, we can just fetch all, assuming idx 0 is safe/padding
                    # But need to handle OOB if map returns None? map.get default 0.
                    
                    target_embs = self.sas_item_emb[idx_array] # (N, H)
                    
                    # Dot product: (N, H) @ (H,) -> (N,)
                    scores = target_embs @ u_emb
                    sasrec_scores = scores
                    has_sas = True

        # 4. Generate Rows
        data = []
        
        # We need to access index for sasrec_scores
        for idx, item in enumerate(candidate_items):
            row = {}
            
            # User Stats
            row['u_cnt'] = u_cnt
            row['u_mean'] = u_mean
            row['u_std'] = u_std
            
            # Item Stats (Lookup O(1))
            i_stat = self.item_stats.get(item, {'count': 0, 'mean': 0, 'std': 0})
            row['i_cnt'] = np.log1p(i_stat['count'])
            row['i_mean'] = i_stat['mean']
            row['i_std'] = i_stat['std'] if not pd.isna(i_stat['std']) else 0
            
            # Content: Complexity
            i_desc_len = self.item_desc_len.get(item, 0)
            row['len_diff'] = abs(u_desc_len - i_desc_len)
            
            # Content: Author
            author = self.item_author.get(item, 'Unknown')
            if (user_id, author) in self.user_author_stats:
                row['u_auth_avg'] = self.user_author_stats[(user_id, author)]
                row['u_auth_match'] = 1
            else:
                row['u_auth_avg'] = u_mean
                row['u_auth_match'] = 0
                
            # SASRec
            row['sasrec_score'] = float(sasrec_scores[idx]) if has_sas else 0.0
            
            # Last-N (Simplify: Skip complex matrix op for now or implement if needed)
            # For strict equivalence, we need loop over history.
            # Leaving as 0.0 for now in Batch unless needed (tradeoff)
            # Or Reuse single logic:
            # (If speed is critical, Last-N is the slowest part usually)
            # Let's call original logic for just this part if SASRec exists? 
            # Or just accept standard python loop for this part.
            
            # Simplified Last-N for batch (placeholder or inline optimization)
            # To properly vectorize Last-N: (N_candidates, H) @ (Last_K_History, H).T -> (N, K) -> max/mean
            
            sim_max, sim_min, sim_mean = 0.0, 0.0, 0.0
            # ... (Vectorized Last-N Implementation) ...
            if has_sas and hasattr(self, 'user_sequences'):
                 # We already have target_embs[idx] from batch step? 
                 # Let's just use the loop logic for Last-N, it's safer.
                 # But efficient: we already fetched u_emb, but we need LAST N items.
                 user_seq = self.user_sequences.get(user_id, [])
                 i_idx_map = self.sasrec_item_map.get(item, 0)
                 if len(user_seq) > 0 and i_idx_map > 0:
                     cand_emb = self.sas_item_emb[i_idx_map]
                     last_n = user_seq[-5:]
                     # This inner loop is 5 iterations. Fast enough.
                     # Optimization: Pre-compute history embeddings for user.
                     pass 
                     # (Actually, let's keep it simple for this file update)
                     
            row['sim_max'] = 0.0 # TODO: Implement if critical
            row['sim_min'] = 0.0
            row['sim_mean'] = 0.0
            
            # Copy logic from generate_features for correctness if not vectorizing everything
            if self.has_sasrec:
                 # Re-use logic for now to ensure correctness
                 feats_single = self.generate_features(user_id, item)
                 row['sim_max'] = feats_single.get('sim_max', 0)
                 row['sim_min'] = feats_single.get('sim_min', 0)
                 row['sim_mean'] = feats_single.get('sim_mean', 0)

            # Category
            cand_cat = self.item_category.get(item, 'Unknown')
            row['is_cat_hob'] = 1 if cand_cat in user_cats else 0
            
            # ItemCF
            icf_score = 0
            icf_max = 0
            
            # Restore definition of itemcf for checking
            itemcf = self.recall_fusion.itemcf
            
            if hasattr(itemcf, "sim_matrix") and item in itemcf.sim_matrix:
                related = itemcf.sim_matrix[item]
                # Intersection of related & history
                # Optimization: iterate smaller set
                common = history.intersection(related.keys())
                if common:
                     sims = [related[c] for c in common]
                     icf_score = sum(sims)
                     icf_max = max(sims)
            
            row['icf_sum'] = icf_score
            row['icf_max'] = icf_max
            
            # UserCF
            ucfscore = 0
            # Loop similar users
            # Need to guard usercf usage? usercf defined above.
            for sim_u, sim_score in usercf_sim_users.items():
                 # Check if sim_u rated this item
                 # Guard: usercf.user_hist might be missing if UserCF failed load or changed
                 if hasattr(usercf, "user_hist") and item in usercf.user_hist.get(sim_u, set()):
                     ucfscore += sim_score
            row['ucf_sum'] = ucfscore
            
            data.append(row)
            
        return pd.DataFrame(data)

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
