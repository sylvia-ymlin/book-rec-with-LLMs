import logging
from collections import defaultdict
from src.recall.itemcf import ItemCF
from src.recall.usercf import UserCF
from src.recall.popularity import PopularityRecall
# from src.recall.embedding import EmbeddingRecall 

logger = logging.getLogger(__name__)

class RecallFusion:
    def __init__(self, data_dir='data/rec', model_dir='data/model/recall'):
        self.itemcf = ItemCF(data_dir, model_dir)
        self.usercf = UserCF(data_dir, model_dir)
        self.popularity = PopularityRecall(data_dir, model_dir)
        # self.embedding = EmbeddingRecall(data_dir, model_dir)
        
        self.models_loaded = False
        
    def load_models(self):
        if self.models_loaded:
            return

        logger.info("Loading recall models...")
        self.itemcf.load()
        self.usercf.load()
        self.popularity.load()
        # self.embedding.load()
        self.models_loaded = True
        
    def get_recall_items(self, user_id, history_items=None, k=100):
        """
        Multi-channel recall fusion using RRF
        """
        if not self.models_loaded:
            self.load_models()
            
        candidates = defaultdict(float)
        
        # 1. ItemCF
        # user_id is mainly used to retrieve training history if history_items is None
        # history_items is passed for realtime inference
        icf_recs = self.itemcf.recommend(user_id, history_items, top_k=k)
        self._add_to_candidates(candidates, icf_recs, weight=1.0)
        
        # 2. UserCF
        # Only works if user_id is in training set
        ucf_recs = self.usercf.recommend(user_id, history_items, top_k=k)
        self._add_to_candidates(candidates, ucf_recs, weight=1.0)
        
        # 3. Popularity (Filler)
        # Only use if candidate set is small? Or always mix in?
        # Let's mix in with lower weight or just use to fill
        pop_recs = self.popularity.recommend(user_id, top_k=k)
        self._add_to_candidates(candidates, pop_recs, weight=0.5)
        
        # Sort by RRF score
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_cands[:k]
    
    def _add_to_candidates(self, candidates, recs, weight=1.0, rrf_k=60):
        """
        Add recommendations to candidate pool using RRF
        score += weight * (1 / (k + rank))
        """
        for rank, (item, score) in enumerate(recs):
            rrf_score = weight * (1.0 / (rrf_k + rank + 1))
            candidates[item] += rrf_score

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    fusion = RecallFusion()
    fusion.load_models()
    
    # Test user
    import pandas as pd
    df = pd.read_csv('data/rec/train.csv')
    user_id = df['user_id'].iloc[0]
    
    recs = fusion.get_recall_items(user_id, k=10)
    print(f"Fused Recs for {user_id}:")
    for item, score in recs:
        print(f"  {item}: {score:.4f}")
