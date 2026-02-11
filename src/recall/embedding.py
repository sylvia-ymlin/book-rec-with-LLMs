"""
YoutubeDNN Two-Tower Recall

V2.7: Replaced torch.matmul brute-force search with Faiss IndexFlatIP
for SIMD-accelerated inner-product retrieval.
"""

import torch
import numpy as np
import pickle
import logging
import faiss
from pathlib import Path
from src.recall.youtube_dnn import YoutubeDNN

logger = logging.getLogger(__name__)

class YoutubeDNNRecall:
    def __init__(self, data_dir='data/rec', model_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # M1/M2 Mac check
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')

        self.model = None
        self.item_vector_index = None # Matrix of item embeddings (torch)
        self.faiss_index = None       # Faiss IndexFlatIP for fast search
        self.item_ids = None # List of item IDs corresponding to rows
        self.user_seqs = {}
        self.item_map = {}
        self.id_to_item = {}
        self.meta = None

    def load(self):
        try:
            logger.info("Loading YoutubeDNN model...")
            # Load metadata
            with open(self.model_dir / 'youtube_dnn_meta.pkl', 'rb') as f:
                self.meta = pickle.load(f)

            # Initialize model
            self.model = YoutubeDNN(
                self.meta['user_config'],
                self.meta['item_config'],
                self.meta['model_config']
            ).to(self.device)

            # Load weights
            # map_location to handle cuda->cpu/mps
            state_dict = torch.load(
                self.model_dir / 'youtube_dnn.pt',
                map_location=self.device
            )
            self.model.load_state_dict(state_dict)
            self.model.eval()

            # Load auxiliary data (Optional/Survival Mode)
            try:
                with open(self.data_dir / 'item_map.pkl', 'rb') as f:
                    self.item_map = pickle.load(f)
                self.id_to_item = {v: k for k, v in self.item_map.items()}
            except Exception as e:
                logger.warning(f"YoutubeDNN: item_map.pkl not found, item-to-id mapping disabled: {e}")
                self.item_map = {}
                self.id_to_item = {}

            try:
                with open(self.data_dir / 'user_sequences.pkl', 'rb') as f:
                    self.user_seqs = pickle.load(f)
            except Exception as e:
                logger.warning(f"YoutubeDNN: user_sequences.pkl not found: {e}")
                self.user_seqs = {}

            # Precompute Item Embeddings + Build Faiss Index
            self._precompute_item_embeddings()

            logger.info("YoutubeDNN loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load YoutubeDNN: {e}")
            return False

    def _precompute_item_embeddings(self):
        logger.info("Precomputing item embeddings for retrieval...")
        vocab_size = self.meta['item_config']['vocab_size']
        item_to_cate = self.meta['item_to_cate']
        default_cate = 1

        # Prepare inputs for all items (excluding padding 0)
        all_items = torch.arange(vocab_size, device=self.device)

        # Build category tensor
        cate_arr = np.full(vocab_size, default_cate, dtype=np.int64)
        for iid, cid in item_to_cate.items():
            if iid < vocab_size:
                cate_arr[iid] = cid
        all_cates = torch.from_numpy(cate_arr).to(self.device)

        # Batch inference
        batch_size = 1024
        vecs_list = []

        with torch.no_grad():
            for i in range(0, vocab_size, batch_size):
                end = min(i + batch_size, vocab_size)
                batch_items = all_items[i:end]
                batch_cates = all_cates[i:end]

                vec = self.model.item_tower(batch_items, batch_cates)
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                vecs_list.append(vec)

        self.item_vector_index = torch.cat(vecs_list, dim=0) # (Vocab, D)
        logger.info(f"Indexed {self.item_vector_index.shape[0]} items.")

        # Build Faiss IndexFlatIP for fast inner-product search
        item_np = self.item_vector_index.cpu().numpy().astype(np.float32)
        item_np = np.ascontiguousarray(item_np)
        dim = item_np.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(item_np)
        logger.info(f"Faiss index built: {self.faiss_index.ntotal} items, dim={dim}")

    def recommend(self, user_id, history_items=None, top_k=50):
        if self.model is None or self.faiss_index is None:
            return []

        # 1. Get User History
        history = []
        if history_items:
            history = [self.item_map.get(isbn, 0) for isbn in history_items]
            history = [x for x in history if x != 0]
        elif self.user_seqs and user_id in self.user_seqs:
            history = self.user_seqs[user_id]

        if not history:
            return []

        # Truncate and Pad
        max_len = self.meta['user_config']['history_len']
        if len(history) > max_len:
            history = history[-max_len:]

        padded_hist = np.zeros(max_len, dtype=np.int64)
        padded_hist[:len(history)] = history

        # 2. Compute User Embedding (still needs torch for model inference)
        hist_tensor = torch.LongTensor(padded_hist).unsqueeze(0).to(self.device)

        with torch.no_grad():
            user_vec = self.model.user_tower(hist_tensor)
            user_vec = torch.nn.functional.normalize(user_vec, p=2, dim=1)

        # 3. Faiss search instead of torch.matmul
        user_np = user_vec.cpu().numpy().astype(np.float32)
        user_np = np.ascontiguousarray(user_np)

        search_k = top_k + len(history) + 10  # oversample for filtering
        scores, indices = self.faiss_index.search(user_np, search_k)
        scores = scores[0]
        indices = indices[0]

        # 4. Map back to ISBNs, filtering padding
        results = []
        for iid, score in zip(indices, scores):
            if iid <= 0:  # skip PAD token at index 0
                continue
            if iid in self.id_to_item:
                isbn = self.id_to_item[iid]
                results.append((isbn, float(score)))
            if len(results) >= top_k:
                break

        return results
