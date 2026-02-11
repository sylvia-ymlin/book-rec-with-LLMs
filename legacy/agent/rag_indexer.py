import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
try:
    from src.data_loader import load_data
except ImportError:
    from data_loader import load_data # Fallback for direct execution

class RAGIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def build_index(self, data: pd.DataFrame):
        """
        Builds the Faiss index from the product dataframe.
        """
        print("Encoding product data...")
        # Create a rich text representation for embedding
        # Title + Description + Features + Category + Price (as text)
        documents = data.apply(lambda x: f"{x['title']} {x['description']} Category: {x['category']} Price: {x['price']}", axis=1).tolist()
        
        embeddings = self.model.encode(documents, show_progress_bar=True)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.metadata = data.to_dict('records')
        print(f"Index built with {len(self.metadata)} items.")

    def save(self, index_path: str, metadata_path: str):
        """
        Saves the index and metadata to disk.
        """
        if self.index:
            faiss.write_index(self.index, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"Saved index to {index_path} and metadata to {metadata_path}")
        else:
            print("No index to save.")

if __name__ == "__main__":
    # Scaffolding run
    df = load_data() # Generates synthetic
    indexer = RAGIndexer()
    indexer.build_index(df)
    
    # Ensure output dir exists
    os.makedirs("data", exist_ok=True)
    indexer.save("data/product_index.faiss", "data/product_metadata.pkl")
