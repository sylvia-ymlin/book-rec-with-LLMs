#!/usr/bin/env python3
"""
Train YoutubeDNN Two-Tower Model for Candidate Retrieval

A deep learning recall model using separate user and item towers.
Trained with in-batch negative sampling for efficient learning.

Usage:
    python scripts/model/train_youtube_dnn.py

Input:
    - data/rec/user_sequences.pkl
    - data/rec/item_map.pkl
    - data/books_processed.csv (for category features)

Output:
    - data/model/recall/youtube_dnn.pt      (model weights)
    - data/model/recall/youtube_dnn_meta.pkl (config + mappings)

Architecture:
    - User Tower: Embedding(history) -> Mean Pooling -> MLP
    - Item Tower: Embedding(item) + Embedding(category) -> MLP
    - Training: Contrastive loss with in-batch negatives

Recommended:
    - GPU: 10-50 epochs, ~30 minutes
    - CPU: 3-5 epochs for testing only
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('.'))
from src.recall.youtube_dnn import YoutubeDNN

# Configuration
BATCH_SIZE = 512
EPOCHS = 10
LR = 0.001
EMBED_DIM = 64
MAX_HISTORY = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

print(f"Using device: {DEVICE}")

def load_data():
    print("Loading data...")
    # Load mappings
    with open('data/rec/item_map.pkl', 'rb') as f:
        item_map = pickle.load(f)
    isbn_to_id = item_map
    id_to_isbn = {v: k for k, v in item_map.items()}
    
    # Load sequences
    with open('data/rec/user_sequences.pkl', 'rb') as f:
        user_seqs = pickle.load(f)
        
    # Load book features for category mapping
    books_df = pd.read_csv('data/books_processed.csv', usecols=['isbn13', 'simple_categories'])
    books_df['isbn'] = books_df['isbn13'].astype(str)
    
    # Create category map
    # Categories are often strings like 'Fiction', 'Juvenile Fiction'. We take the first one.
    cate_map = {'<PAD>': 0, '<UNK>': 1}
    item_to_cate = {}
    
    print("Building category map...")
    for _, row in books_df.iterrows():
        isbn = row['isbn']
        if isbn in isbn_to_id:
            iid = isbn_to_id[isbn]
            cates = str(row['simple_categories']).split(';')
            main_cate = cates[0].strip() if cates else 'Unknown'
            
            if main_cate not in cate_map:
                cate_map[main_cate] = len(cate_map)
            
            item_to_cate[iid] = cate_map[main_cate]
            
    # Default category for unknown items
    default_cate = cate_map.get('Unknown', 1)
    
    return user_seqs, item_to_cate, len(item_map)+1, len(cate_map), default_cate

class RetrievalDataset(Dataset):
    def __init__(self, user_seqs, item_to_cate, default_cate, max_history=20):
        self.samples = []
        self.item_to_cate = item_to_cate
        self.default_cate = default_cate
        self.max_history = max_history
        
        print("Generating training samples...")
        # Leave-Last-Out:
        # Last item -> Test
        # 2nd Last -> Val
        # Rest -> Train
        # So we use items 0 to N-3 for training history generation
        
        for user, seq in tqdm(user_seqs.items()):
            if len(seq) < 3:
                continue
            
            # Use data up to the split point for training
            # Valid Train Set: seq[:-2]
            train_seq = seq[:-2]
            
            # Generate sliding window samples
            # minimum history length = 1
            for i in range(1, len(train_seq)):
                target = train_seq[i]
                history = train_seq[:i]
                
                # Truncate history
                if len(history) > max_history:
                    history = history[-max_history:]
                
                self.samples.append((history, target))
                
        print(f"Total training samples: {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        history, target = self.samples[idx]
        
        # Padding history
        padded_hist = np.zeros(self.max_history, dtype=np.int64)
        length = min(len(history), self.max_history)
        if length > 0:
            padded_hist[:length] = history[-length:]
            
        target_cate = self.item_to_cate.get(target, self.default_cate)
        
        return torch.LongTensor(padded_hist), torch.tensor(target, dtype=torch.long), torch.tensor(target_cate, dtype=torch.long)

def train():
    user_seqs, item_to_cate, vocab_size, cate_vocab_size, default_cate = load_data()
    
    dataset = RetrievalDataset(user_seqs, item_to_cate, default_cate, MAX_HISTORY)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # mp issue on mac sometimes
    
    # Model Setup
    user_config = {
        'vocab_size': vocab_size,
        'embed_dim': EMBED_DIM,
        'history_len': MAX_HISTORY
    }
    item_config = {
        'vocab_size': vocab_size,
        'embed_dim': EMBED_DIM,
        'cate_vocab_size': cate_vocab_size,
        'cate_embed_dim': 32
    }
    model_config = {
        'hidden_dims': [128, 64],
        'dropout': 0.1
    }
    
    model = YoutubeDNN(user_config, item_config, model_config).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss() # For In-Batch Negatives
    
    print("Start Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0
        
        pbar = tqdm(dataloader)
        for history, target_item, target_cate in pbar:
            history = history.to(DEVICE)
            target_item = target_item.to(DEVICE)
            target_cate = target_cate.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Get Vectors
            user_vec = model.user_tower(history) # (B, D)
            item_vec = model.item_tower(target_item, target_cate) # (B, D)
            
            # Normalize
            user_vec = nn.functional.normalize(user_vec, p=2, dim=1)
            item_vec = nn.functional.normalize(item_vec, p=2, dim=1)
            
            # In-Batch Negatives
            # logits[i][j] = user_i dot item_j
            # We want logits[i][i] to be high
            logits = torch.matmul(user_vec, item_vec.t()) # (B, B)
            
            # Temperature scaling (optional, helps convergence)
            logits = logits / 0.1 
            
            labels = torch.arange(len(user_vec)).to(DEVICE)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            pbar.set_description(f"Epoch {epoch+1} Loss: {total_loss/steps:.4f}")
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/steps:.4f}")
        
    # Save Model
    save_path = Path('data/model/recall')
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_path / 'youtube_dnn.pt')
    
    # Save metadata
    meta = {
        'user_config': user_config,
        'item_config': item_config,
        'model_config': model_config,
        'item_to_cate': item_to_cate
    }
    with open(save_path / 'youtube_dnn_meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
        
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()
