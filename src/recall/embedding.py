import torch
import numpy as np
import pickle
import logging
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
        self.item_vector_index = None # Matrix of item embeddings
        self.item_ids = None # List of item IDs corresponding to rows
        self.user_seqs = None
        self.item_map = None
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
            
            # Load auxiliary data
            with open(self.data_dir / 'item_map.pkl', 'rb') as f:
                self.item_map = pickle.load(f)
            self.id_to_item = {v: k for k, v in self.item_map.items()}
            
            with open(self.data_dir / 'user_sequences.pkl', 'rb') as f:
                self.user_seqs = pickle.load(f)
                
            # Precompute Item Embeddings
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
        # We can just iterate 1..vocab_size-1
        all_items = torch.arange(vocab_size, device=self.device)
        
        # Build category tensor
        # Can be optimized but simple loop is fine for once
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

    def recommend(self, user_id, history_items=None, top_k=50):
        if self.model is None:
            return []
            
        # 1. Get User History
        history = []
        if history_items:
            # Real-time history derived from input
            # Convert isbns to ids
            history = [self.item_map.get(isbn, 0) for isbn in history_items]
            history = [x for x in history if x != 0]
        elif user_id in self.user_seqs:
            # Offline history
            history = self.user_seqs[user_id]
        
        if not history:
            return []
            
        # Truncate and Pad
        max_len = self.meta['user_config']['history_len']
        if len(history) > max_len:
            history = history[-max_len:]
            
        padded_hist = np.zeros(max_len, dtype=np.int64)
        padded_hist[:len(history)] = history
        
        # 2. Compute User Embedding
        hist_tensor = torch.LongTensor(padded_hist).unsqueeze(0).to(self.device) # (1, L)
        
        with torch.no_grad():
            user_vec = self.model.user_tower(hist_tensor) # (1, D)
            user_vec = torch.nn.functional.normalize(user_vec, p=2, dim=1)
            
            # 3. Dot Product Search
            # (1, D) @ (Vocab, D).T = (1, Vocab)
            scores = torch.matmul(user_vec, self.item_vector_index.t()).squeeze(0) # (Vocab,)
            
            # Mask special tokens/history?
            scores[0] = -float('inf') # Mask PAD
            
            # Filter history items? usually yes
            # for hid in history:
            #     scores[hid] = -float('inf')
                
            # Top K
            top_scores, top_indices = torch.topk(scores, k=top_k)
            
        # 4. Map back to ISBNs
        results = []
        top_indices = top_indices.cpu().numpy()
        top_scores = top_scores.cpu().numpy()
        
        for iid, score in zip(top_indices, top_scores):
            if iid in self.id_to_item:
                isbn = self.id_to_item[iid]
                results.append((isbn, float(score)))
                
        return results
