"""
SASRec Embedding Recall

Uses pre-trained SASRec user sequence embeddings and item embeddings
to perform dot-product based candidate retrieval.

V2.7: Replaced numpy brute-force dot-product with Faiss IndexFlatIP
for SIMD-accelerated approximate nearest neighbor search.
"""

import pickle
import logging
import numpy as np
import faiss
from pathlib import Path

logger = logging.getLogger(__name__)


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

    def load(self):
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
