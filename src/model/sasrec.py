import torch
import torch.nn as nn
import numpy as np

class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, hidden_dim=64, num_blocks=2, num_heads=2, dropout_rate=0.2):
        super(SASRec, self).__init__()
        
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        
        # Embeddings
        # Item embedding (0 is padding)
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        # Positional embedding
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        
        self.emb_dropout = nn.Dropout(dropout_rate)
        
        # Transformer Blocks
        # Standard PyTorch TransformerEncoder is easiest
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        
        self.last_layernorm = nn.LayerNorm(hidden_dim)
        
    def forward(self, input_seqs):
        # input_seqs: [batch_size, max_len]
        batch_size = input_seqs.shape[0]
        device = input_seqs.device
        
        # 1. Generate Masks
        # Padding mask: Ignore 0s
        padding_mask = (input_seqs == 0) # [batch, len]
        
        # Causal mask: Future positions cannot see current
        # [len, len]
        sz = input_seqs.shape[1]
        # triu=1 means upper triangle is 1 (masked). 
        # But PyTorch transformer src_mask: -inf for masked, 0 for allowed.
        causal_mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        
        # 2. Embedding
        seqs = self.item_emb(input_seqs) # [batch, len, dim]
        seqs = seqs * (self.hidden_dim ** 0.5) # Scale
        
        # Add position
        positions = torch.arange(sz, device=device).unsqueeze(0).repeat(batch_size, 1)
        # Positions should only be valid for non-padding?
        # Standard SASRec uses absolute input position.
        pos_emb = self.pos_emb(positions)
        
        x = seqs + pos_emb
        x = self.emb_dropout(x)
        
        # 3. Transformer
        # src_key_padding_mask: True for padded positions
        output = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        output = self.last_layernorm(output)
        
        return output # [batch, len, dim]

    def predict(self, input_seqs, candidate_items):
        # input_seqs: [batch, len]
        # candidate_items: [batch] (Single item) or [batch, num_candidates]
        
        # We only care about the specific output at the LAST valid position
        # But for batch efficiency in training, we usually use full sequence.
        
        # For inference (which is what we care about for features):
        # Get the embedding of the LAST token in the sequence
        
        seq_output = self.forward(input_seqs)
        
        # Take the last element: [batch, dim]
        # Note: If padding exists at end (which shouldn't happen with our left-pad logic, 
        # but actually we usually pad at LEFT or RIGHT? 
        # SASRec standard is usually Pad Left? Or just simple pad.
        # Let's assume input_seqs is valid till end or find last non-zero.
        
        # Simplified: take last hidden state
        final_state = seq_output[:, -1, :] 
        
        return final_state
