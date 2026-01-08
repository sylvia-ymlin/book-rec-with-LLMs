import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from src.model.sasrec import SASRec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeqDataset(Dataset):
    def __init__(self, seqs_dict, num_items, max_len=50):
        self.seqs = []
        self.num_items = num_items
        
        # Prepare (seq_in, target) pairs
        for u, s in seqs_dict.items():
            if len(s) < 2:
                continue
            
            # Pad
            seq_processed = [0] * max_len
            seq_len = min(len(s), max_len)
            seq_processed[-seq_len:] = s[-seq_len:]
            
            self.seqs.append(seq_processed)
            
        self.seqs = torch.LongTensor(self.seqs)
        
    def __len__(self):
        return len(self.seqs)
        
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        
        # Determine pos/neg for training
        # Target: seq shifted right (positives)
        pos = np.zeros_like(seq)
        pos[:-1] = seq[1:]
        
        # Negatives: random sample
        neg = np.random.randint(1, self.num_items + 1, size=len(seq))
        
        return seq, torch.LongTensor(pos), torch.LongTensor(neg)

def train_sasrec():
    max_len = 50
    hidden_dim = 64
    batch_size = 128
    epochs = 3
    lr = 0.001
    
    data_dir = Path('data/rec')
    logger.info("Loading sequences...")
    with open(data_dir / 'user_sequences.pkl', 'rb') as f:
        seqs_dict = pickle.load(f)
        
    with open(data_dir / 'item_map.pkl', 'rb') as f:
        item_map = pickle.load(f)
    num_items = len(item_map)
    
    dataset = SeqDataset(seqs_dict, num_items, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Check for MPS (Mac GPU) or CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    logger.info(f"Training on {device}")
    
    model = SASRec(num_items, max_len, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # BCE Loss for Pos/Neg
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for seq, pos, neg in pbar:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            
            # Forward pass to get seq embeddings: [B, L, H]
            seq_emb = model(seq) # [B, L, H]
            
            # Mask padding (0) in targets
            # pos has 0 where padding exists (shifted)
            # But we should use original seq mask.
            # Only calculate loss for non-zero targets
            mask = (pos != 0)
            
            # Get Item Embeddings for Pos and Neg
            pos_emb = model.item_emb(pos) # [B, L, H]
            neg_emb = model.item_emb(neg) # [B, L, H]
            
            # Calculate logits (Dot Product)
            # [B, L, H] * [B, L, H] -> [B, L]
            pos_logits = (seq_emb * pos_emb).sum(dim=-1)
            neg_logits = (seq_emb * neg_emb).sum(dim=-1)
            
            # Loss
            # Pos -> 1, Neg -> 0
            # Apply mask to flat vectors
            
            pos_logits = pos_logits[mask]
            neg_logits = neg_logits[mask]
            
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            
            loss = criterion(pos_logits, pos_labels) + criterion(neg_logits, neg_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
            
    # Save Model
    torch.save(model.state_dict(), data_dir / '../model/rec/sasrec_model.pth')
    
    # Save User Embeddings (Last hidden state)
    logger.info("Extracting User Sequence Embeddings...")
    model.eval()
    user_emb_dict = {}
    
    # Create evaluation loader (no shuffle, keep user order)
    # We need to map back to user_ids, so iterate dict directly
    
    # Batch processing for inference
    all_users = list(seqs_dict.keys())
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_users), batch_size)):
            batch_users = all_users[i : i+batch_size]
            batch_seqs = []
            for u in batch_users:
                s = seqs_dict[u]
                # Same padding logic
                seq_processed = [0] * max_len
                seq_len = min(len(s), max_len)
                if seq_len > 0:
                    seq_processed[-seq_len:] = s[-seq_len:]
                batch_seqs.append(seq_processed)
                
            input_tensor = torch.LongTensor(batch_seqs).to(device)
            
            # Initial forward
            # Note: During inference, we use the FULL sequence to predict the FUTURE (Test Item)
            # So input is the full available history
            
            output = model(input_tensor) # [B, L, H]
            last_state = output[:, -1, :].cpu().numpy() # [B, H]
            
            for j, u in enumerate(batch_users):
                user_emb_dict[u] = last_state[j]
                
    with open(data_dir / 'user_seq_emb.pkl', 'wb') as f:
        pickle.dump(user_emb_dict, f)
        
    logger.info("User Seq Embeddings saved.")

if __name__ == "__main__":
    train_sasrec()
