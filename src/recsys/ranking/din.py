"""
DIN (Deep Interest Network) ranker.

Moved from `src/ranking/din.py` into `recsys.ranking`.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class DIN(nn.Module):
    """
    Deep Interest Network: attention over user history w.r.t. target item.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        max_hist_len: int = 50,
        mlp_dims: tuple = (128, 64, 32),
        dropout: float = 0.1,
        num_aux: int = 0,
        pretrained_item_emb: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_hist_len = max_hist_len
        self.num_aux = num_aux

        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)

        if pretrained_item_emb is not None:
            self._init_from_pretrained(pretrained_item_emb)

        self.attn_fc = nn.Sequential(
            nn.Linear(embed_dim * 4, 36),
            nn.ReLU(),
            nn.Linear(36, 1),
        )

        mlp_in = embed_dim * 2 + num_aux
        layers = []
        for d in mlp_dims:
            layers.append(nn.Linear(mlp_in, d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            mlp_in = d
        layers.append(nn.Linear(mlp_in, 1))
        self.mlp = nn.Sequential(*layers)

    def _init_from_pretrained(self, emb: np.ndarray) -> None:
        """Initialize item_emb from SASRec checkpoint."""
        if emb.shape[0] >= self.num_items + 1 and emb.shape[1] == self.embed_dim:
            with torch.no_grad():
                self.item_emb.weight.data[: emb.shape[0]].copy_(torch.from_numpy(emb))
            logger.info(
                "DIN: Initialized item_emb from pretrained (%d x %d)", *emb.shape
            )
        else:
            logger.warning(
                "DIN: Pretrained shape %s mismatch, skipping init", emb.shape
            )

    def forward(
        self,
        user_hist: torch.Tensor,
        target_item: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hist_embs = self.item_emb(user_hist)
        target_emb = self.item_emb(target_item)

        target_expand = target_emb.unsqueeze(1).expand(-1, user_hist.size(1), -1)
        attn_input = torch.cat(
            [
                hist_embs,
                target_expand,
                hist_embs * target_expand,
                hist_embs - target_expand,
            ],
            dim=-1,
        )
        attn_scores = self.attn_fc(attn_input).squeeze(-1)

        mask = (user_hist != 0).float()
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn_weights * mask
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)

        user_interest = (hist_embs * attn_weights.unsqueeze(-1)).sum(dim=1)

        mlp_in = torch.cat([user_interest, target_emb], dim=1)
        if aux_features is not None and self.num_aux > 0 and aux_features.size(-1) == self.num_aux:
            mlp_in = torch.cat([mlp_in, aux_features], dim=1)

        logits = self.mlp(mlp_in).squeeze(-1)
        return logits


from src.recsys.ranking.base import BaseRanker


class DINRanker(BaseRanker):
    """
    Wrapper for DIN model: load, predict, compatible with RecommendationService.
    """

    def __init__(
        self,
        data_dir: str = "data/rec",
        model_dir: str = "data/model",
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir) / "ranking"
        self.model: Optional[DIN] = None
        self.item_map: dict = {}
        self.id_to_item: dict = {}
        self.user_sequences: dict = {}
        self.max_hist_len = 50
        self.aux_feature_names: list = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

    @property
    def feature_names(self) -> list[str]:
        return self.aux_feature_names

    def load(self) -> bool:
        """Load trained DIN and aux data."""
        import pickle

        model_path = self.model_dir / "din_ranker.pt"
        if not model_path.exists():
            return False

        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = ckpt["model"]
            self.model.to(self.device)
            self.model.eval()
            self.item_map = ckpt.get("item_map", {})
            self.id_to_item = {v: k for k, v in self.item_map.items()}
            self.max_hist_len = ckpt.get("max_hist_len", 50)
            self.aux_feature_names = ckpt.get("aux_feature_names", [])

            with open(self.data_dir / "user_sequences.pkl", "rb") as f:
                seqs = pickle.load(f)
            self.user_sequences = seqs

            logger.info("DIN ranker loaded from %s", model_path)
            return True
        except Exception as e:
            logger.error("Failed to load DIN ranker: %s", e)
            return False

    def predict(
        self,
        user_id: str,
        candidate_items: list[str],
        features_df: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict scores for (user_id, candidate_items).
        """
        override_hist = kwargs.get("override_hist")
        aux_features = None
        if features_df is not None and self.aux_feature_names:
            # Extract aux features from features_df if provided
            aux_features = features_df[self.aux_feature_names].values.astype(np.float32)
        if self.model is None:
            self.load()
        if self.model is None:
            return np.zeros(len(candidate_items))

        hist = (
            override_hist
            if override_hist is not None
            else self.user_sequences.get(user_id, [])
        )
        if hist and isinstance(hist[0], str):
            hist = [self.item_map.get(h, 0) for h in hist]
        hist = hist[-self.max_hist_len:]
        padded = np.zeros(self.max_hist_len, dtype=np.int64)
        padded[: len(hist)] = hist

        target_ids = np.array(
            [self.item_map.get(str(it), 0) for it in candidate_items],
            dtype=np.int64,
        )

        hist_t = (
            torch.LongTensor(padded)
            .unsqueeze(0)
            .expand(len(candidate_items), -1)
            .to(self.device)
        )
        target_t = torch.LongTensor(target_ids).to(self.device)

        aux_t = None
        if aux_features is not None and aux_features.size > 0:
            aux_t = torch.from_numpy(aux_features.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits = self.model(hist_t, target_t, aux_t)
        return logits.cpu().numpy()


__all__ = ["DIN", "DINRanker"]

