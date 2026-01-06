import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class ProductRetriever:
    def __init__(self, index_path: str, metadata_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Searches for the top-k most relevant products.
        """
        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                item = self.metadata[idx]
                item['score'] = float(distances[0][i])
                results.append(item)
                
        return results

if __name__ == "__main__":
    # Test run
    retriever = ProductRetriever("data/product_index.faiss", "data/product_metadata.pkl")
    query = "cheap gaming laptop"
    results = retriever.search(query)
    
    print(f"Query: {query}")
    for res in results:
        print(f" - {res['title']} (${res['price']}) [Score: {res['score']:.4f}]")
