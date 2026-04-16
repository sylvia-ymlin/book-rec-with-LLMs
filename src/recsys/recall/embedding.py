"""
YoutubeDNN Two-Tower Recall.

Moved from `src/recall/embedding.py` into the `recsys.recall` package.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.recsys.recall.sequence_utils import build_sequences_from_df
from src.recsys.recall.youtube_dnn import YoutubeDNN


logger = logging.getLogger(__name__)


class _RetrievalDataset(Dataset):
    """Internal dataset for YoutubeDNN training (history -> target)."""

    def __init__(
        self,
        user_seqs: dict,
        item_to_cate: dict,
        default_cate: int,
        max_history: int,
    ):
        self.samples: list[tuple[list[int], int, int]] = []
        self.item_to_cate = item_to_cate
        self.default_cate = default_cate
        self.max_history = max_history

        for _user, seq in user_seqs.items():
            if len(seq) < 3:
                continue
            train_seq = seq[:-2]
            for i in range(1, len(train_seq)):
                target = train_seq[i]
                history = train_seq[:i]
                if len(history) > max_history:
                    history = history[-max_history:]
                target_cate = item_to_cate.get(target, default_cate)
                self.samples.append((history, target, target_cate))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history, target, target_cate = self.samples[idx]
        padded = np.zeros(self.max_history, dtype=np.int64)
        length = min(len(history), self.max_history)
        if length > 0:
            padded[:length] = history[-length:]
        return (
            torch.LongTensor(padded),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(target_cate, dtype=torch.long),
        )


class YoutubeDNNRecall:
    def __init__(self, data_dir="data/rec", model_dir="data/model/recall"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

        self.model: Optional[YoutubeDNN] = None
        self.item_vector_index = None
        self.faiss_index = None
        self.item_ids = None
        self.user_seqs: dict = {}
        self.item_map: dict = {}
        self.id_to_item: dict = {}
        self.meta: Optional[dict] = None

    def fit(
        self,
        df: pd.DataFrame,
        books_path: Optional[Path] = None,
        epochs: int = 10,
        batch_size: int = 512,
        lr: float = 0.001,
        embed_dim: int = 64,
        max_history: int = 20,
    ) -> "YoutubeDNNRecall":
        logger.info("Building sequences from DataFrame...")
        user_seqs, item_map = build_sequences_from_df(df, max_len=50)

        self.item_map = item_map
        self.id_to_item = {v: k for k, v in item_map.items()}
        vocab_size = len(item_map) + 1

        cate_map: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        item_to_cate: dict[int, int] = {}
        default_cate = 1

        books_path = (
            Path(books_path)
            if books_path
            else self.data_dir.parent / "books_processed.csv"
        )
        if books_path.exists():
            books_df = pd.read_csv(
                books_path, usecols=["isbn13", "simple_categories"]
            )
            books_df["isbn"] = books_df["isbn13"].astype(str)
            for _, row in books_df.iterrows():
                isbn = str(row["isbn"])
                if isbn in item_map:
                    iid = item_map[isbn]
                    cates = str(row["simple_categories"]).split(";")
                    main_cate = cates[0].strip() if cates else "Unknown"
                    if main_cate not in cate_map:
                        cate_map[main_cate] = len(cate_map)
                    item_to_cate[iid] = cate_map[main_cate]
            default_cate = cate_map.get("Unknown", 1)

        for iid in range(1, vocab_size):
            if iid not in item_to_cate:
                item_to_cate[iid] = default_cate

        cate_vocab_size = len(cate_map)

        user_config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "history_len": max_history,
        }
        item_config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "cate_vocab_size": cate_vocab_size,
            "cate_embed_dim": 32,
        }
        model_config = {"hidden_dims": [128, 64], "dropout": 0.1}

        dataset = _RetrievalDataset(
            user_seqs, item_to_cate, default_cate, max_history
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        self.model = YoutubeDNN(
            user_config, item_config, model_config
        ).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        logger.info("Training YoutubeDNN...")
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for history, target_item, target_cate in tqdm(
                dataloader, desc=f"Epoch {epoch+1}"
            ):
                history = history.to(self.device)
                target_item = target_item.to(self.device)
                target_cate = target_cate.to(self.device)
                optimizer.zero_grad()
                user_vec = self.model.user_tower(history)
                item_vec = self.model.item_tower(target_item, target_cate)
                user_vec = nn.functional.normalize(user_vec, p=2, dim=1)
                item_vec = nn.functional.normalize(item_vec, p=2, dim=1)
                logits = torch.matmul(user_vec, item_vec.t()) / 0.1
                labels = torch.arange(len(user_vec)).to(self.device)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            logger.info(
                "Epoch %d Loss: %.4f", epoch + 1, total_loss / max(steps, 1)
            )

        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_dir / "youtube_dnn.pt")
        self.meta = {
            "user_config": user_config,
            "item_config": item_config,
            "model_config": model_config,
            "item_to_cate": item_to_cate,
        }
        with open(self.model_dir / "youtube_dnn_meta.pkl", "wb") as f:
            pickle.dump(self.meta, f)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.data_dir / "item_map.pkl", "wb") as f:
            pickle.dump(self.item_map, f)
        with open(self.data_dir / "user_sequences.pkl", "wb") as f:
            pickle.dump(user_seqs, f)

        logger.info("YoutubeDNN saved to %s", self.model_dir)
        return self

    def _precompute_item_embeddings(self):
        logger.info("Precomputing item embeddings for retrieval...")
        vocab_size = self.meta["item_config"]["vocab_size"]
        item_to_cate = self.meta["item_to_cate"]
        default_cate = 1

        all_items = torch.arange(vocab_size, device=self.device)

        cate_arr = np.full(vocab_size, default_cate, dtype=np.int64)
        for iid, cid in item_to_cate.items():
            if iid < vocab_size:
                cate_arr[iid] = cid
        all_cates = torch.from_numpy(cate_arr).to(self.device)

        batch_size = 1024
        vecs_list = []

        with torch.no_grad():
            for i in range(0, vocab_size, batch_size):
                end = min(i + batch_size, vocab_size)
                batch_items = all_items[i:end]
                batch_cates = all_cates[i:end]

                vec = self.model.item_tower(batch_items, batch_cates)
                vec = nn.functional.normalize(vec, p=2, dim=1)
                vecs_list.append(vec)

        self.item_vector_index = torch.cat(vecs_list, dim=0)
        logger.info("Indexed %d items.", self.item_vector_index.shape[0])

        item_np = self.item_vector_index.cpu().numpy().astype(np.float32)
        item_np = np.ascontiguousarray(item_np)
        dim = item_np.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(item_np)
        logger.info(
            "Faiss index built: %d items, dim=%d",
            self.faiss_index.ntotal,
            dim,
        )

    def load(self) -> bool:
        try:
            logger.info("Loading YoutubeDNN model...")
            with open(self.model_dir / "youtube_dnn_meta.pkl", "rb") as f:
                self.meta = pickle.load(f)

            self.model = YoutubeDNN(
                self.meta["user_config"],
                self.meta["item_config"],
                self.meta["model_config"],
            ).to(self.device)

            state_dict = torch.load(
                self.model_dir / "youtube_dnn.pt", map_location=self.device
            )
            # Remap old named-layer keys (user_fc_0, item_fc_0 …) to the
            # current nn.Sequential integer-index keys (0, 3, …).
            # Old format: user_mlp.user_fc_0.weight → new: user_mlp.0.weight
            #             user_mlp.user_fc_1.weight → new: user_mlp.3.weight
            #             item_mlp.item_fc_0.weight → new: item_mlp.0.weight
            #             item_mlp.item_fc_1.weight → new: item_mlp.3.weight
            _KEY_MAP = {
                "user_mlp.user_fc_0.": "user_mlp.0.",
                "user_mlp.user_fc_1.": "user_mlp.3.",
                "item_mlp.item_fc_0.": "item_mlp.0.",
                "item_mlp.item_fc_1.": "item_mlp.3.",
            }
            remapped = {}
            for k, v in state_dict.items():
                new_k = k
                for old_prefix, new_prefix in _KEY_MAP.items():
                    if k.startswith(old_prefix):
                        new_k = new_prefix + k[len(old_prefix):]
                        break
                remapped[new_k] = v
            if remapped.keys() != state_dict.keys():
                logger.info("YoutubeDNN: remapped %d legacy state-dict keys.",
                            sum(1 for k in state_dict if k != remapped.get(k, k)))
                state_dict = remapped
            self.model.load_state_dict(state_dict)
            self.model.eval()

            try:
                with open(self.data_dir / "item_map.pkl", "rb") as f:
                    self.item_map = pickle.load(f)
                self.id_to_item = {
                    v: k for k, v in self.item_map.items()
                }
            except Exception as e:
                logger.warning(
                    "YoutubeDNN: item_map.pkl not found, item-to-id mapping disabled: %s",
                    e,
                )
                self.item_map = {}
                self.id_to_item = {}

            try:
                with open(self.data_dir / "user_sequences.pkl", "rb") as f:
                    self.user_seqs = pickle.load(f)
            except Exception as e:
                logger.warning(
                    "YoutubeDNN: user_sequences.pkl not found: %s", e
                )
                self.user_seqs = {}

            self._precompute_item_embeddings()

            logger.info("YoutubeDNN loaded successfully.")
            return True
        except Exception as e:
            logger.error("Failed to load YoutubeDNN: %s", e)
            return False

    def recommend(self, user_id, history_items=None, top_k=50):
        if self.model is None or self.faiss_index is None:
            return []

        history: list[int] = []
        if history_items:
            history = [self.item_map.get(isbn, 0) for isbn in history_items]
            history = [x for x in history if x != 0]
        elif self.user_seqs and user_id in self.user_seqs:
            history = self.user_seqs[user_id]

        if not history:
            return []

        max_len = self.meta["user_config"]["history_len"]
        if len(history) > max_len:
            history = history[-max_len:]

        padded_hist = np.zeros(max_len, dtype=np.int64)
        padded_hist[: len(history)] = history

        hist_tensor = torch.LongTensor(padded_hist).unsqueeze(0).to(self.device)

        with torch.no_grad():
            user_vec = self.model.user_tower(hist_tensor)
            user_vec = nn.functional.normalize(user_vec, p=2, dim=1)

        user_np = user_vec.cpu().numpy().astype(np.float32)
        user_np = np.ascontiguousarray(user_np)

        search_k = top_k + len(history) + 10
        scores, indices = self.faiss_index.search(user_np, search_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for iid, score in zip(indices, scores):
            if iid <= 0:
                continue
            if iid in self.id_to_item:
                isbn = self.id_to_item[iid]
                results.append((isbn, float(score)))
            if len(results) >= top_k:
                break

        return results


__all__ = ["YoutubeDNNRecall"]

