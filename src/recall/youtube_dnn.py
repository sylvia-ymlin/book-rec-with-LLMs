
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class YoutubeDNN(nn.Module):
    def __init__(self, user_config, item_config, model_config):
        """
        YoutubeDNN Model (Dual Tower)
        
        Args:
            user_config (dict): {
                'vocab_size': int, # Number of items for history embedding
                'embed_dim': int,  # Dimension of item embeddings
                'history_len': int # Max history length
            }
            item_config (dict): {
                'vocab_size': int, # Number of items (same as user_config vocab)
                'embed_dim': int,  # Dimension of item embeddings
                'cate_vocab_size': int, # Number of categories
                'cate_embed_dim': int # Dimension of category embeddings
            }
            model_config (dict): {
                'hidden_dims': list, # [256, 128, 64]
                'dropout': float
            }
        """
        super(YoutubeDNN, self).__init__()
        
        # Shared Item Enbedding (used for both History and Item Tower target)
        # In typical YoutubeDNN, the Softmax weights are the item embeddings.
        # But for Dual Tower (DSSM style), we have separate towers.
        # Here we share the item embedding matrix for history lookups and item tower input.
        self.item_embedding = nn.Embedding(
            item_config['vocab_size'], 
            item_config['embed_dim'], 
            padding_idx=0
        )
        
        # --- User Tower ---
        # Input: History Item Embeddings (Mean/Sum)
        self.user_layer_dims = [item_config['embed_dim']] + model_config['hidden_dims']
        
        self.user_mlp = nn.Sequential()
        for i in range(len(self.user_layer_dims) - 1):
            self.user_mlp.add_module(f'user_fc_{i}', nn.Linear(self.user_layer_dims[i], self.user_layer_dims[i+1]))
            self.user_mlp.add_module(f'user_relu_{i}', nn.ReLU())
            self.user_mlp.add_module(f'user_drop_{i}', nn.Dropout(model_config['dropout']))
            
        # --- Item Tower ---
        # Input: Item Embedding + Category Embedding
        self.cate_embedding = nn.Embedding(
            item_config['cate_vocab_size'],
            item_config['cate_embed_dim'],
            padding_idx=0
        )
        
        self.item_input_dim = item_config['embed_dim'] + item_config['cate_embed_dim']
        self.item_layer_dims = [self.item_input_dim] + model_config['hidden_dims']
        
        self.item_mlp = nn.Sequential()
        for i in range(len(self.item_layer_dims) - 1):
            self.item_mlp.add_module(f'item_fc_{i}', nn.Linear(self.item_layer_dims[i], self.item_layer_dims[i+1]))
            self.item_mlp.add_module(f'item_relu_{i}', nn.ReLU())
            self.item_mlp.add_module(f'item_drop_{i}', nn.Dropout(model_config['dropout']))
            
    def user_tower(self, history_ids):
        """
        Args:
            history_ids: (batch_size, history_len) - LongTensor
        """
        # Mask for padding (0)
        mask = (history_ids != 0).unsqueeze(-1).float() # (B, L, 1)
        
        # Embeddings
        embeds = self.item_embedding(history_ids) # (B, L, E)
        
        # Masked Mean Pooling
        sum_embeds = (embeds * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        mean_embeds = sum_embeds / count # (B, E)
        
        # MLP
        user_vec = self.user_mlp(mean_embeds)
        return user_vec
        
    def item_tower(self, item_ids, cate_ids):
        """
        Args:
            item_ids: (batch_size,)
            cate_ids: (batch_size,)
        """
        item_embed = self.item_embedding(item_ids) # (B, E)
        cate_embed = self.cate_embedding(cate_ids) # (B, Ec)
        
        # Concat
        concat = torch.cat([item_embed, cate_embed], dim=1) # (B, E+Ec)
        
        # MLP
        item_vec = self.item_mlp(concat)
        return item_vec
        
    def forward(self, history_ids, target_item_ids, target_cate_ids):
        """
        Training forward pass.
        Returns: logits (dot product)
        Note: This returns simple dot product. For training, we usually want
        positives and negatives, or CrossEntropy over all items.
        Here we assume we compute dot product for the pair.
        """
        user_vec = self.user_tower(history_ids) # (B, D)
        item_vec = self.item_tower(target_item_ids, target_cate_ids) # (B, D)
        
        # Normalize vectors for cosine similarity (optional but good for retrieval)
        user_vec = F.normalize(user_vec, p=2, dim=1)
        item_vec = F.normalize(item_vec, p=2, dim=1)
        
        # Dot product
        score = (user_vec * item_vec).sum(dim=1)
        return score 

