"""
Item2Vec Recall: Word2Vec-based item embedding similarity.

Treats user interaction sequences as "sentences" and items (ISBNs) as "words".
Trains Word2Vec (Skip-gram) to learn item embeddings, then builds a similarity
matrix for fast retrieval.

Reference: Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for
Collaborative Filtering", 2016.
"""

import pickle
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


class Item2Vec:
    def __init__(self, data_dir='data/rec', save_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sim_matrix = {}
        self.user_hist = {}

    def fit(self, df, vector_size=64, window=5, min_count=3, sg=1, epochs=10, top_k_sim=200):
        """
        Train Item2Vec embeddings and build similarity matrix.

        Phase 1: Build ISBN-based user sequences sorted by timestamp.
        Phase 2: Train Word2Vec (Skip-gram) on sequences.
        Phase 3: Build sim_matrix from learned embeddings.

        Args:
            df: DataFrame with [user_id, isbn, rating, timestamp]
            vector_size: embedding dimension (64 to match SASRec)
            window: Word2Vec context window
            min_count: minimum item frequency to include
            sg: 1=Skip-gram, 0=CBOW
            epochs: Word2Vec training epochs
            top_k_sim: keep top-k similar items per item
        """
        logger.info("Building Item2Vec embeddings...")

        # 1. Build user -> items mapping (for recommend())
        user_items = defaultdict(set)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building index"):
            user_items[row['user_id']].add(row['isbn'])
        self.user_hist = {u: items for u, items in user_items.items()}

        # 2. Build "sentences" = user interaction sequences sorted by timestamp
        #    Each sentence is a list of ISBN strings (Word2Vec treats them as tokens)
        logger.info("Building interaction sequences...")
        df_sorted = df.sort_values(['user_id', 'timestamp'])
        sentences = []
        for user_id, group in df_sorted.groupby('user_id'):
            seq = group['isbn'].tolist()
            if len(seq) >= 2:  # need at least 2 items to form context
                sentences.append(seq)

        logger.info(f"Built {len(sentences)} sequences for Word2Vec training")

        # 3. Train Word2Vec
        logger.info(f"Training Word2Vec (dim={vector_size}, window={window}, "
                     f"sg={sg}, epochs={epochs})...")
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=4,
            epochs=epochs,
            seed=42,
        )
        vocab_items = list(model.wv.index_to_key)
        logger.info(f"Word2Vec trained: {len(vocab_items)} items in vocabulary")

        # 4. Build similarity matrix: for each item, find top-k most similar
        #    gensim most_similar() returns cosine similarity in [-1, 1],
        #    but top similar items will have positive cosine — no renormalization needed.
        logger.info("Building similarity matrix from embeddings...")
        final_sim = {}
        for item in tqdm(vocab_items, desc="Computing similarities"):
            try:
                similar = model.wv.most_similar(item, topn=top_k_sim)
                final_sim[item] = {sim_item: score for sim_item, score in similar}
            except KeyError:
                continue

        self.sim_matrix = final_sim
        self.save()
        logger.info(f"Item2Vec matrix built: {len(final_sim)} items")
        return self.sim_matrix

    def recommend(self, user_id, history_items=None, top_k=50):
        """
        Recommend items based on embedding similarity to user history.
        Sum cosine similarity from each history item to candidate.
        """
        rank = defaultdict(float)

        if history_items is None:
            if user_id in self.user_hist:
                history_items = list(self.user_hist[user_id])
            else:
                return []

        history_set = set(history_items)

        for item_i in history_items:
            if item_i in self.sim_matrix:
                for item_j, score in self.sim_matrix[item_i].items():
                    if item_j in history_set:
                        continue
                    rank[item_j] += score

        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save(self):
        with open(self.save_dir / 'item2vec.pkl', 'wb') as f:
            pickle.dump({
                'sim_matrix': self.sim_matrix,
                'user_hist': self.user_hist
            }, f)
        logger.info(f"Item2Vec model saved to {self.save_dir / 'item2vec.pkl'}")

    def load(self):
        path = self.save_dir / 'item2vec.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.sim_matrix = data['sim_matrix']
                self.user_hist = data['user_hist']
            logger.info(f"Item2Vec model loaded from {path}")
            return True
        return False


if __name__ == "__main__":
    import pandas as pd
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv('data/rec/train.csv')

    model = Item2Vec()
    model.fit(df)

    # Test rec
    user_id = df['user_id'].iloc[0]
    recs = model.recommend(user_id)
    print(f"Recs for {user_id}: {recs[:5]}")
