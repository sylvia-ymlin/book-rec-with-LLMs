"""
SASRec Embedding Recall

Uses pre-trained SASRec user sequence embeddings and item embeddings
to perform dot-product based candidate retrieval.

V2.7: Replaced numpy brute-force dot-product with Faiss IndexFlatIP
for SIMD-accelerated approximate nearest neighbor search.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model.sasrec import SASRec
from src.recall.sequence_utils import build_sequences_from_df

logger = logging.getLogger(__name__)


class _SeqDataset(Dataset):
    """Internal dataset for SASRec training (seq, pos, neg)."""

    def __init__(self, seqs_dict: dict, num_items: int, max_len: int):
        self.seqs: list[list[int]] = []
        self.num_items = num_items
        self.max_len = max_len

        for seq in seqs_dict.values():
            if len(seq) < 2:
                continue
            padded = [0] * max_len
            seq_len = min(len(seq), max_len)
            padded[-seq_len:] = seq[-seq_len:]
            self.seqs.append(padded)
        self.seqs = torch.LongTensor(self.seqs)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.seqs[idx]
        pos = np.zeros_like(seq.numpy())
        pos[:-1] = seq.numpy()[1:]
        neg = np.random.randint(1, self.num_items + 1, size=len(seq))
        return seq, torch.LongTensor(pos), torch.LongTensor(neg)


class SASRecRecall:
    def __init__(self, data_dir='data/rec', model_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

        self.user_seq_emb = {}   # user_id -> np.array (embedding)
        self.item_emb = None     # np.array [num_items+1, dim]
        self.item_map = {}       # isbn -> item_index
        self.id_to_item = {}     # item_index -> isbn
        self.user_hist = {}      # user_id -> set of isbns (for filtering)
        self.faiss_index = None  # Faiss IndexFlatIP for fast inner-product search
        self.loaded = False

    def fit(
        self,
        df: pd.DataFrame,
        max_len: int = 50,
        hidden_dim: int = 64,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 1e-4,
    ) -> "SASRecRecall":
        """
        Train SASRec from interaction DataFrame. Builds sequences internally.

        Args:
            df: [user_id, isbn, timestamp] (timestamp optional)
            max_len, hidden_dim, epochs, batch_size, lr: Training hyperparameters.
        """
        logger.info("Building sequences from DataFrame...")
        user_seqs, item_map = build_sequences_from_df(df, max_len=max_len)

        self.item_map = item_map
        self.id_to_item = {v: k for k, v in item_map.items()}
        num_items = len(item_map)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logger.info(f"Training SASRec on {device}...")
        dataset = _SeqDataset(user_seqs, num_items, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = SASRec(num_items, max_len, hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for seq, pos, neg in pbar:
                seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
                seq_emb = model(seq)
                mask = pos != 0
                pos_emb = model.item_emb(pos)
                neg_emb = model.item_emb(neg)
                pos_logits = (seq_emb * pos_emb).sum(dim=-1)[mask]
                neg_logits = (seq_emb * neg_emb).sum(dim=-1)[mask]
                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)
                loss = criterion(pos_logits, pos_labels) + criterion(neg_logits, neg_labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        # Save model
        sasrec_dir = self.model_dir.parent / "rec"
        sasrec_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), sasrec_dir / "sasrec_model.pth")

        # Extract user embeddings
        logger.info("Extracting user sequence embeddings...")
        model.eval()
        user_emb_dict: dict = {}
        all_users = list(user_seqs.keys())

        with torch.no_grad():
            for i in tqdm(range(0, len(all_users), batch_size), desc="Embedding users"):
                batch_users = all_users[i : i + batch_size]
                batch_seqs = []
                for u in batch_users:
                    s = user_seqs[u]
                    padded = [0] * max_len
                    seq_len = min(len(s), max_len)
                    if seq_len > 0:
                        padded[-seq_len:] = s[-seq_len:]
                    batch_seqs.append(padded)
                input_tensor = torch.LongTensor(batch_seqs).to(device)
                output = model(input_tensor)
                last_state = output[:, -1, :].cpu().numpy()
                for j, u in enumerate(batch_users):
                    user_emb_dict[u] = last_state[j]

        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.data_dir / "user_seq_emb.pkl", "wb") as f:
            pickle.dump(user_emb_dict, f)
        with open(self.data_dir / "item_map.pkl", "wb") as f:
            pickle.dump(self.item_map, f)
        with open(self.data_dir / "user_sequences.pkl", "wb") as f:
            pickle.dump(user_seqs, f)

        self.user_seq_emb = user_emb_dict
        self.user_hist = {
            u: set(self.id_to_item[idx] for idx in seq if idx in self.id_to_item)
            for u, seq in user_seqs.items()
        }
        self.item_emb = model.item_emb.weight.detach().cpu().numpy()
        self._build_faiss_index()
        self.loaded = True

        logger.info(f"SASRec saved to {sasrec_dir}")
        return self

    def _build_faiss_index(self) -> None:
        """Build Faiss index from item embeddings."""
        if self.item_emb is None:
            return
        dim = self.item_emb.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(np.ascontiguousarray(self.item_emb.astype(np.float32)))

    def load(self) -> bool:
        try:
            logger.info("Loading SASRec recall embeddings...")

            # 1. User sequence embeddings (pre-computed)
            with open(self.data_dir / 'user_seq_emb.pkl', 'rb') as f:
                self.user_seq_emb = pickle.load(f)

            # 2. Item map
            with open(self.data_dir / 'item_map.pkl', 'rb') as f:
                self.item_map = pickle.load(f)
            self.id_to_item = {v: k for k, v in self.item_map.items()}

            # 3. Item embeddings from SASRec model checkpoint
            import torch
            model_path = self.model_dir.parent / 'rec' / 'sasrec_model.pth'
            state_dict = torch.load(model_path, map_location='cpu')
            self.item_emb = state_dict['item_emb.weight'].numpy()  # [N+1, dim]

            # 4. Build Faiss IndexFlatIP for fast inner-product search
            dim = self.item_emb.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            item_emb_f32 = np.ascontiguousarray(self.item_emb.astype(np.float32))
            self.faiss_index.add(item_emb_f32)
            logger.info(f"Faiss index built: {self.faiss_index.ntotal} items, dim={dim}")

            # 5. User history for filtering
            try:
                with open(self.data_dir / 'user_sequences.pkl', 'rb') as f:
                    user_seqs = pickle.load(f)
                # Convert item indices back to ISBNs for filtering
                self.user_hist = {}
                for uid, seq in user_seqs.items():
                    self.user_hist[uid] = set(
                        self.id_to_item[idx] for idx in seq if idx in self.id_to_item
                    )
            except Exception as e:
                logger.warning(f"SASRec: user_sequences.pkl not found: {e}")
                self.user_hist = {}

            self.loaded = True
            logger.info(f"SASRec recall loaded: {len(self.user_seq_emb)} users, {self.item_emb.shape[0]} items")
            return True

        except Exception as e:
            logger.warning(f"Failed to load SASRec recall: {e}")
            self.loaded = False
            return False

    def recommend(self, user_id, history_items=None, top_k=50):
        if not self.loaded or self.faiss_index is None:
            return []

        # Get user embedding
        u_emb = self.user_seq_emb.get(user_id)
        if u_emb is None:
            return []

        # Build history mask
        history_set = set()
        if history_items:
            history_set = set(history_items)
        elif user_id in self.user_hist:
            history_set = self.user_hist[user_id]

        # Faiss search (inner product)
        query = np.ascontiguousarray(u_emb.reshape(1, -1).astype(np.float32))
        search_k = top_k + len(history_set) + 10  # oversample for filtering
        scores, indices = self.faiss_index.search(query, search_k)
        scores = scores[0]   # (search_k,)
        indices = indices[0]  # (search_k,)

        # Filter and collect results
        results = []
        for idx, score in zip(indices, scores):
            if idx <= 0:  # skip padding index 0 and invalid -1
                continue
            isbn = self.id_to_item.get(int(idx))
            if isbn is None:
                continue
            if isbn in history_set:
                continue
            results.append((isbn, float(score)))
            if len(results) >= top_k:
                break

        return results
