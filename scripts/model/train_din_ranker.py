#!/usr/bin/env python3
"""
Train DIN (Deep Interest Network) ranker.

Uses attention over user behavior sequence w.r.t. target item.
Reuses SASRec item embeddings as initialization when available.

Usage:
    python scripts/model/train_din_ranker.py
    python scripts/model/train_din_ranker.py --max_samples 10000 --epochs 10

Input:
    - data/rec/val.csv, train.csv
    - data/rec/user_sequences.pkl, item_map.pkl (from SASRec/YoutubeDNN)
    - data/model/rec/sasrec_model.pth (optional, for init)

Output:
    - data/model/ranking/din_ranker.pt
"""

import sys
import os

sys.path.append(os.getcwd())

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.ranking.din import DIN
from src.recall.fusion import RecallFusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_din_data(
    data_dir: str = "data/rec",
    model_dir: str = "data/model/recall",
    neg_ratio: int = 4,
    max_samples: int = 20000,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Build (user_id, isbn, label) samples with hard negatives.
    Returns (df, user_sequences, item_map).
    """
    logger.info("Building DIN training data...")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    all_items = pd.read_csv(f"{data_dir}/train.csv")["isbn"].astype(str).unique()

    if len(val_df) > max_samples:
        val_df = val_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    fusion = RecallFusion(data_dir, model_dir)
    fusion.load_models()

    with open(f"{data_dir}/user_sequences.pkl", "rb") as f:
        user_sequences = pickle.load(f)
    with open(f"{data_dir}/item_map.pkl", "rb") as f:
        item_map = pickle.load(f)

    rows = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Mining samples"):
        user_id = str(row["user_id"])
        pos_isbn = str(row["isbn"])

        user_rows = [{"user_id": user_id, "isbn": pos_isbn, "label": 1}]

        try:
            recall_items = fusion.get_recall_items(user_id, k=50)
            hard_negs = [item for item, _ in recall_items if item != pos_isbn][:neg_ratio]
        except Exception:
            hard_negs = []

        for neg_isbn in hard_negs:
            user_rows.append({"user_id": user_id, "isbn": str(neg_isbn), "label": 0})

        n_remaining = neg_ratio - len(hard_negs)
        if n_remaining > 0:
            random_negs = np.random.choice(all_items, size=n_remaining, replace=False)
            for neg_isbn in random_negs:
                user_rows.append({"user_id": user_id, "isbn": str(neg_isbn), "label": 0})

        rows.extend(user_rows)

    df = pd.DataFrame(rows)
    logger.info(f"Built {len(df)} samples")
    return df, user_sequences, item_map


class DINDataset(Dataset):
    """Dataset for DIN: (user_hist, target_item_id, label) and optional aux features."""

    def __init__(
        self,
        df: pd.DataFrame,
        user_sequences: dict,
        item_map: dict,
        max_hist_len: int = 50,
        aux_df: pd.DataFrame | None = None,
        aux_cols: list[str] | None = None,
    ):
        self.samples = []
        self.aux_df = aux_df
        self.aux_cols = aux_cols or []
        for idx, (_, row) in enumerate(df.iterrows()):
            user_id = str(row["user_id"])
            isbn = str(row["isbn"])
            label = int(row["label"])
            target_id = item_map.get(isbn, 0)
            if target_id == 0:
                continue

            hist = user_sequences.get(user_id, [])
            if hist and isinstance(hist[0], str):
                hist = [item_map.get(h, 0) for h in hist if item_map.get(h, 0) > 0]
            hist = [x for x in hist if x != target_id][-max_hist_len:]

            self.samples.append((hist, target_id, label, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        hist, target_id, label, df_idx = self.samples[idx]
        max_len = 50
        padded = np.zeros(max_len, dtype=np.int64)
        padded[: len(hist)] = hist
        out = (
            torch.LongTensor(padded),
            torch.LongTensor([target_id]).squeeze(0),
            torch.FloatTensor([label]).squeeze(0),
        )
        if self.aux_df is not None and self.aux_cols:
            aux_row = self.aux_df.iloc[df_idx][self.aux_cols].values.astype(np.float32)
            out = out + (torch.FloatTensor(aux_row),)
        return out


def train_din(
    data_dir: str = "data/rec",
    model_dir: str = "data/model",
    recall_dir: str = "data/model/recall",
    max_samples: int = 20000,
    max_hist_len: int = 50,
    embed_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    use_aux: bool = False,
) -> None:
    rank_dir = Path(model_dir) / "ranking"
    rank_dir.mkdir(parents=True, exist_ok=True)

    df, user_sequences, item_map = build_din_data(
        data_dir, recall_dir, neg_ratio=4, max_samples=max_samples
    )
    num_items = len(item_map)

    aux_df = None
    aux_cols: list[str] = []
    if use_aux:
        from src.ranking.features import FeatureEngineer
        fe = FeatureEngineer(data_dir, recall_dir)
        fe.load_base_data()
        logger.info("Generating aux features for DIN...")
        aux_df = fe.create_dateset(df)
        aux_cols = [c for c in aux_df.columns if c not in ("label", "user_id", "isbn")]
        logger.info("Aux features: %s", aux_cols)

    num_aux = len(aux_cols)
    dataset = DINDataset(
        df, user_sequences, item_map,
        max_hist_len=max_hist_len,
        aux_df=aux_df,
        aux_cols=aux_cols if aux_cols else None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    pretrained_emb = None
    sasrec_path = Path(model_dir) / "rec" / "sasrec_model.pth"
    if sasrec_path.exists():
        try:
            state = torch.load(sasrec_path, map_location="cpu", weights_only=False)
            emb = state.get("item_emb.weight")
            if emb is not None:
                pretrained_emb = emb.numpy()
                logger.info("Loaded SASRec item_emb for DIN init: %s", pretrained_emb.shape)
        except Exception as e:
            logger.warning("Could not load SASRec init: %s", e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    model = DIN(
        num_items=num_items,
        embed_dim=embed_dim,
        max_hist_len=max_hist_len,
        num_aux=num_aux,
        pretrained_item_emb=pretrained_emb,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(loader, desc=f"Epoch {ep+1}/{epochs}"):
            hist = batch[0].to(device)
            target = batch[1].to(device)
            label = batch[2].to(device)
            aux = batch[3].to(device) if len(batch) > 3 else None
            opt.zero_grad()
            logits = model(hist, target, aux)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        logger.info(f"Epoch {ep+1} loss: {avg:.4f}")

    ckpt = {
        "model": model,
        "item_map": item_map,
        "max_hist_len": max_hist_len,
        "aux_feature_names": aux_cols,
    }
    out_path = rank_dir / "din_ranker.pt"
    torch.save(ckpt, out_path)
    logger.info("DIN ranker saved to %s", out_path)

    with open(Path(data_dir) / "user_sequences.pkl", "wb") as f:
        pickle.dump(user_sequences, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DIN ranker")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--aux", action="store_true", help="Use aux features from FeatureEngineer")
    args = parser.parse_args()

    train_din(
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_aux=args.aux,
    )
