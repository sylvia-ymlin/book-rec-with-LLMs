
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('.'))
from src.recall.youtube_dnn import YoutubeDNN

# Config
BATCH_SIZE = 128
EMBED_DIM = 64
MAX_HISTORY = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

print(f"Using device: {DEVICE}")

def load_resources():
    print("Loading resources...")
    base_path = Path('data/rec')
    model_path = Path('data/model/recall')
    
    # Mappings
    with open(base_path / 'item_map.pkl', 'rb') as f:
        item_map = pickle.load(f)
    item_to_id = item_map
    id_to_item = {v: k for k, v in item_map.items()}
    
    # Sequences
    with open(base_path / 'user_sequences.pkl', 'rb') as f:
        user_seqs = pickle.load(f)
        
    # Metadata
    with open(model_path / 'youtube_dnn_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
        
    return item_to_id, id_to_item, user_seqs, meta

def get_item_features(item_to_id, item_to_cate, default_cate, vocab_size):
    """Prepare tensor for all items"""
    # Create tensors for all item IDs [0, vocab_size-1]
    # 0 is PAD
    # We want to score items 1 to vocab_size-1
    
    all_items = torch.arange(vocab_size, device=DEVICE)
    all_cates = []
    
    # Map items to cates
    # Can be slow if loop, let's vectorise or list comp
    # item_to_cate is dict {iid: cate_id}
    
    # Precompute array
    cate_arr = np.full(vocab_size, default_cate, dtype=np.int64)
    for iid, cid in item_to_cate.items():
        if iid < vocab_size:
            cate_arr[iid] = cid
            
    all_cates = torch.from_numpy(cate_arr).to(DEVICE)
    
    return all_items, all_cates

def evaluate_model(sample_n=500):
    item_to_id, id_to_item, user_seqs, meta = load_resources()
    
    # Load Model
    print("Loading model...")
    model = YoutubeDNN(
        meta['user_config'],
        meta['item_config'],
        meta['model_config']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load('data/model/recall/youtube_dnn.pt', map_location=DEVICE))
    model.eval()
    
    # Load Test Data
    test_df = pd.read_csv('data/rec/test.csv')
    if sample_n:
        test_df = test_df.sample(n=sample_n, random_state=42)
        
    print(f"Evaluating on {len(test_df)} users...")
    
    # Precompute All Item Embeddings
    print("Precomputing item embeddings...")
    vocab_size = meta['item_config']['vocab_size']
    item_to_cate = meta['item_to_cate']
    # Infer default cate from meta or heuristics (usually 1 for UNK)
    default_cate = 1
    
    all_items, all_cates = get_item_features(item_to_id, item_to_cate, default_cate, vocab_size)
    
    # Batch Compute Item Embeddings to save GPU memory
    item_vecs = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, vocab_size, batch_size):
            end = min(i + batch_size, vocab_size)
            batch_items = all_items[i:end]
            batch_cates = all_cates[i:end]
            
            vec = model.item_tower(batch_items, batch_cates)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            item_vecs.append(vec)
            
    all_item_vecs = torch.cat(item_vecs, dim=0) # (Vocab, D)
    print(f"Item Embeddings Shape: {all_item_vecs.shape}")
    
    # Evaluate Loop
    hits_10 = 0
    hits_50 = 0
    mrr_10 = 0
    
    # Prepare User Batches
    user_ids_list = test_df['user_id'].tolist()
    target_isbns_list = test_df['isbn'].tolist()
    
    # Process one by one (or batch)
    # Since history length varies, let's do one by one or collate
    
    print("Computing metrics...")
    
    # Cache lookup
    target_item_ids = []
    histories = []
    valid_indices = []
    
    for idx, (uid, isbn) in enumerate(zip(user_ids_list, target_isbns_list)):
        if isbn not in item_to_id:
            continue
            
        target_iid = item_to_id[isbn]
        
        if uid not in user_seqs:
            continue
            
        seq = user_seqs[uid]
        # Test phase history: all items except the last one (which is the target)
        # But wait, user_seqs contains ALL items including test item?
        # Let's check split logic.
        # process.py usually groups all interactions.
        # split_rec_data does not modify user_sequences.pkl directly, it just reads raw df.
        # user_sequences.pkl was likely created earlier containing ALL interactions?
        # Let's assume user_seqs has the full sequence including the test item at the end.
        
        # Test target is indeed the last item in time sorted sequence.
        # So history is seq[:-1]
        
        history = seq[:-1]
        if not history:
            continue
            
        # Truncate
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
            
        # Pad
        padded = np.zeros(MAX_HISTORY, dtype=np.int64)
        padded[:len(history)] = history
        
        histories.append(padded)
        target_item_ids.append(target_iid)
        valid_indices.append(idx)
        
    # To Tensor
    hist_tensor = torch.LongTensor(np.array(histories)).to(DEVICE) # (N, L)
    
    # Compute User Vecs
    user_vecs = []
    with torch.no_grad():
        for i in range(0, len(hist_tensor), BATCH_SIZE):
            batch_hist = hist_tensor[i:i+BATCH_SIZE]
            u_vec = model.user_tower(batch_hist)
            u_vec = torch.nn.functional.normalize(u_vec, p=2, dim=1)
            user_vecs.append(u_vec)
            
    all_user_vecs = torch.cat(user_vecs, dim=0) # (N, D)
    
    # Compute Scores: User x Item
    # (N, D) @ (Vocab, D).T = (N, Vocab)
    # This might be large.
    # We can compute row by row if needed.
    
    print(f"Scoring {len(all_user_vecs)} users against {vocab_size} items...")
    
    metrics = {
        'hit_10': 0,
        'hit_50': 0,
        'mrr_10': 0
    }
    
    # Chunking similarity computation to avoid OOM
    # If vocab=200k, floats=4bytes. 200k*4 = 800KB items. 
    # N=500. 500 * 200k * 4 = 400MB. Steps fit easily in memory.
    
    scores = torch.matmul(all_user_vecs, all_item_vecs.t()) # (N, Vocab)
    
    # For each user, find rank of target
    # We can use topk
    
    # Mask out special tokens (0, 1 etc) if needed?
    # Usually item IDs start from 1. 0 is padding.
    scores[:, 0] = -float('inf') # Mask padding
    
    _, top_indices = torch.topk(scores, k=50, dim=1) # (N, 50)
    
    top_indices = top_indices.cpu().numpy()
    targets = np.array(target_item_ids)
    
    for i in range(len(targets)):
        target = targets[i]
        top_items = top_indices[i]
        
        if target in top_items:
            rank = np.where(top_items == target)[0][0]
            metrics['hit_50'] += 1
            
            if rank < 10:
                metrics['hit_10'] += 1
                metrics['mrr_10'] += 1.0 / (rank + 1)
                
    n_eval = len(targets)
    print("\n" + "="*30)
    print(f"RESULTS (Sample N={n_eval})")
    print("="*30)
    print(f"Hit Rate@10: {metrics['hit_10']/n_eval:.4f}")
    print(f"Hit Rate@50: {metrics['hit_50']/n_eval:.4f}")
    print(f"MRR@10:      {metrics['mrr_10']/n_eval:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_model(sample_n=1000)
