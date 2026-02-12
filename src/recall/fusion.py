"""
Multi-Channel Recall Fusion via Reciprocal Rank Fusion (RRF).

ALGORITHM (RRF - Cormack et al., 2009):
  When merging ranked lists from multiple recall channels (ItemCF, SASRec, etc.),
  each channel contributes: score += weight * 1 / (k + rank).
  rank = 0-based position in that channel's list. k = RRF_K (default 60).

  Intuition: Rank 1 contributes more than rank 10. Larger k = less sensitive to
  exact rank position (smoother fusion). k=60 is a common default from TREC.

CHANNELS:
  Controlled by channel_config (enable/disable, weight per channel).
  Default: itemcf, sasrec, youtube_dnn, popularity. Others (usercf, swing, item2vec) off.
"""
import logging
from collections import defaultdict
from typing import Optional

from src.config import RRF_K
from src.recall.itemcf import ItemCF
from src.recall.usercf import UserCF
from src.recall.popularity import PopularityRecall
from src.recall.embedding import YoutubeDNNRecall
from src.recall.swing import Swing
from src.recall.item2vec import Item2Vec
from src.recall.sasrec_recall import SASRecRecall

logger = logging.getLogger(__name__)

# Default: only the 3 most effective channels enabled. Others available but off.
DEFAULT_CHANNEL_CONFIG = {
    "itemcf": {"enabled": True, "weight": 1.0},
    "sasrec": {"enabled": True, "weight": 1.0},
    "youtube_dnn": {"enabled": True, "weight": 1.0},
    "usercf": {"enabled": False, "weight": 1.0},
    "swing": {"enabled": False, "weight": 1.0},
    "item2vec": {"enabled": False, "weight": 0.8},
    "popularity": {"enabled": True, "weight": 0.5},  # P0: Cold-start fallback
}


def _merge_config(default: dict, override: Optional[dict]) -> dict:
    """Deep-merge override into default (shallow per channel)."""
    merged = {k: dict(v) for k, v in default.items()}
    if override:
        for ch, cfg in override.items():
            if ch in merged:
                merged[ch].update(cfg)
            else:
                merged[ch] = dict(cfg)
    return merged


class RecallFusion:
    def __init__(
        self,
        data_dir: str = "data/rec",
        model_dir: str = "data/model/recall",
        channel_config: Optional[dict] = None,
        rrf_k: int = RRF_K,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.channel_config = _merge_config(DEFAULT_CHANNEL_CONFIG, channel_config)
        self.rrf_k = rrf_k

        self.itemcf = ItemCF(data_dir, model_dir)
        self.usercf = UserCF(data_dir, model_dir)
        self.popularity = PopularityRecall(data_dir, model_dir)
        self.youtube_dnn = YoutubeDNNRecall(data_dir, model_dir)
        self.swing = Swing(data_dir, model_dir)
        self.item2vec = Item2Vec(data_dir, model_dir)
        self.sasrec = SASRecRecall(data_dir, model_dir)

        self.models_loaded = False

    def load_models(self) -> None:
        if self.models_loaded:
            return

        logger.info("Loading recall models...")
        self.itemcf.load()
        self.usercf.load()
        self.popularity.load()
        self.youtube_dnn.load()
        self.swing.load()
        self.item2vec.load()
        self.sasrec.load()
        self.models_loaded = True

    def get_recall_items(
        self,
        user_id: str,
        history_items=None,
        k: int = 100,
        real_time_seq=None,
    ):
        """
        Multi-channel recall fusion using RRF. Channels and weights controlled by config.

        Args:
            real_time_seq: P1 - Session-level ISBNs to inject into SASRec (e.g. just-viewed).
        """
        if not self.models_loaded:
            self.load_models()

        candidates = defaultdict(float)
        cfg = self.channel_config

        if cfg.get("youtube_dnn", {}).get("enabled", False):
            recs = self.youtube_dnn.recommend(user_id, history_items, top_k=k)
            self._add_to_candidates(candidates, recs, cfg["youtube_dnn"]["weight"])

        if cfg.get("itemcf", {}).get("enabled", False):
            recs = self.itemcf.recommend(user_id, history_items, top_k=k)
            self._add_to_candidates(candidates, recs, cfg["itemcf"]["weight"])

        if cfg.get("usercf", {}).get("enabled", False):
            recs = self.usercf.recommend(user_id, history_items, top_k=k)
            self._add_to_candidates(candidates, recs, cfg["usercf"]["weight"])

        if cfg.get("swing", {}).get("enabled", False):
            recs = self.swing.recommend(user_id, history_items, top_k=k)
            self._add_to_candidates(candidates, recs, cfg["swing"]["weight"])

        if cfg.get("sasrec", {}).get("enabled", False):
            recs = self.sasrec.recommend(
                user_id, history_items, top_k=k, real_time_seq=real_time_seq
            )
            self._add_to_candidates(candidates, recs, cfg["sasrec"]["weight"])

        if cfg.get("item2vec", {}).get("enabled", False):
            recs = self.item2vec.recommend(user_id, history_items, top_k=k)
            self._add_to_candidates(candidates, recs, cfg["item2vec"]["weight"])

        if cfg.get("popularity", {}).get("enabled", False):
            recs = self.popularity.recommend(user_id, top_k=k)
            self._add_to_candidates(candidates, recs, cfg["popularity"]["weight"])

        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        # P0: Cold-start fallback — when all channels return empty, use popularity
        if not sorted_cands:
            pop_recs = self.popularity.recommend(user_id, top_k=k)
            sorted_cands = [(item, s) for item, s in pop_recs]
        return sorted_cands[:k]

    def _add_to_candidates(self, candidates, recs, weight: float) -> None:
        """
        RRF: score += weight * 1/(k + rank + 1).
        rank is 0-based. Items appearing in multiple channels get summed scores.
        """
        if not recs:
            return
        for rank, (item, score) in enumerate(recs):
            rrf_score = weight * (1.0 / (self.rrf_k + rank + 1))
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
