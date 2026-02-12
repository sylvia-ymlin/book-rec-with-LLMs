"""
YoutubeDNN dual-tower model.

Moved from `src/recall/youtube_dnn.py` into `recsys.recall`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YoutubeDNN(nn.Module):
    def __init__(self, user_config, item_config, model_config):
        """
        YoutubeDNN Model (Dual Tower)

        Args:
            user_config (dict): {
                'vocab_size': int,  # Number of items for history embedding
                'embed_dim': int,   # Dimension of item embeddings
                'history_len': int  # Max history length
            }
            item_config (dict): {
                'vocab_size': int,      # Number of items (same as user_config vocab)
                'embed_dim': int,       # Dimension of item embeddings
                'cate_vocab_size': int, # Number of categories
                'cate_embed_dim': int   # Dimension of category embeddings
            }
            model_config (dict): {
                'hidden_dims': list,  # [256, 128, 64]
                'dropout': float
            }
        """
        super().__init__()

        self.item_embedding = nn.Embedding(
            item_config["vocab_size"],
            item_config["embed_dim"],
            padding_idx=0,
        )

        self.user_layer_dims = [item_config["embed_dim"]] + model_config[
            "hidden_dims"
        ]
        user_layers = []
        for i in range(len(self.user_layer_dims) - 1):
            user_layers.append(
                nn.Linear(
                    self.user_layer_dims[i], self.user_layer_dims[i + 1]
                )
            )
            user_layers.append(nn.ReLU())
            user_layers.append(nn.Dropout(model_config["dropout"]))
        self.user_mlp = nn.Sequential(*user_layers)

        self.cate_embedding = nn.Embedding(
            item_config["cate_vocab_size"],
            item_config["cate_embed_dim"],
            padding_idx=0,
        )

        self.item_input_dim = (
            item_config["embed_dim"] + item_config["cate_embed_dim"]
        )
        self.item_layer_dims = [self.item_input_dim] + model_config[
            "hidden_dims"
        ]
        item_layers = []
        for i in range(len(self.item_layer_dims) - 1):
            item_layers.append(
                nn.Linear(
                    self.item_layer_dims[i], self.item_layer_dims[i + 1]
                )
            )
            item_layers.append(nn.ReLU())
            item_layers.append(nn.Dropout(model_config["dropout"]))
        self.item_mlp = nn.Sequential(*item_layers)

    def user_tower(self, history_ids):
        """
        Args:
            history_ids: (batch_size, history_len) - LongTensor
        """
        mask = (history_ids != 0).unsqueeze(-1).float()
        embeds = self.item_embedding(history_ids)

        sum_embeds = (embeds * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        mean_embeds = sum_embeds / count

        user_vec = self.user_mlp(mean_embeds)
        return user_vec

    def item_tower(self, item_ids, cate_ids):
        """
        Args:
            item_ids: (batch_size,)
            cate_ids: (batch_size,)
        """
        item_embed = self.item_embedding(item_ids)
        cate_embed = self.cate_embedding(cate_ids)

        concat = torch.cat([item_embed, cate_embed], dim=1)
        item_vec = self.item_mlp(concat)
        return item_vec

    def forward(self, history_ids, target_item_ids, target_cate_ids):
        """
        Training forward pass.
        Returns: logits (dot product)
        """
        user_vec = self.user_tower(history_ids)
        item_vec = self.item_tower(target_item_ids, target_cate_ids)

        user_vec = F.normalize(user_vec, p=2, dim=1)
        item_vec = F.normalize(item_vec, p=2, dim=1)

        score = (user_vec * item_vec).sum(dim=1)
        return score


__all__ = ["YoutubeDNN"]

