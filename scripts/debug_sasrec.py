import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from src.model.sasrec import SASRec
from torch.utils.data import DataLoader, Dataset

# Mini Dataset
class SeqDataset(Dataset):
    def __init__(self, seqs_dict, num_items, max_len=50):
        self.seqs = []
        self.num_items = num_items
        for u, s in seqs_dict.items():
            if len(s) < 2: continue
            seq_processed = [0] * max_len
            seq_len = min(len(s), max_len)
            seq_processed[-seq_len:] = s[-seq_len:]
            self.seqs.append(seq_processed)
        self.seqs = torch.LongTensor(self.seqs)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        pos = np.zeros_like(seq)
        pos[:-1] = seq[1:]
        neg = np.random.randint(1, self.num_items + 1, size=len(seq))
        return seq, torch.LongTensor(pos), torch.LongTensor(neg)

def debug():
    data_dir = Path('data/rec')
    with open(data_dir / 'user_sequences.pkl', 'rb') as f:
        seqs_dict = pickle.load(f)
    with open(data_dir / 'item_map.pkl', 'rb') as f:
        item_map = pickle.load(f)
    num_items = len(item_map)
    print(f"Num items: {num_items}")

    # Check max id
    max_id = 0
    for s in seqs_dict.values():
        if s: max_id = max(max_id, max(s))
    print(f"Max Item ID in data: {max_id}")
    
    dataset = SeqDataset(seqs_dict, num_items)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SASRec(num_items, 50, 64)
    criterion = nn.BCEWithLogitsLoss()
    
    batch = next(iter(loader))
    seq, pos, neg = batch
    
    print("Input range:", seq.min(), seq.max())
    print("Pos range:", pos.min(), pos.max())
    print("Neg range:", neg.min(), neg.max())
    
    seq_emb = model(seq)
    if torch.isnan(seq_emb).any():
        print("NaN in seq_emb!")
    
    pos_emb = model.item_emb(pos)
    neg_emb = model.item_emb(neg)
    
    pos_logits = (seq_emb * pos_emb).sum(dim=-1)
    neg_logits = (seq_emb * neg_emb).sum(dim=-1)
    
    print("Logits sample:", pos_logits[0])
    
    loss = criterion(pos_logits, torch.ones_like(pos_logits))
    print(f"Loss: {loss.item()}")
    
if __name__ == "__main__":
    debug()
