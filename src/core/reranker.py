from typing import List, Tuple, Dict, Any
from sentence_transformers import CrossEncoder
import torch
from src.utils import setup_logger

logger = setup_logger(__name__)

# 轻量级重排序模型，速度快且效果不错
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class RerankerService:
    """
    Singleton service for re-ranking documents using a Cross-Encoder.
    This significantly improves RAG precision by scoring the exact relevance
    of (query, document) pairs.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance
    
    def __init__(self):
        if self.model is None:
            self._load_model()
            
    def _load_model(self):
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Loading Reranker model: {DEFAULT_RERANKER_MODEL} on {device}...")
            self.model = CrossEncoder(DEFAULT_RERANKER_MODEL, device=device)
            logger.info("Reranker model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Reranker: {e}")
            self.model = None

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on relevance to the query.
        
        Args:
            query: User question
            docs: List of dicts, each must have a 'content' field (or 'description')
            top_k: Number of results to return
            
        Returns:
            Top-K sorted documents with added 'score' field.
        """
        if not self.model or not docs:
            return docs[:top_k]
            
        # Prepare pairs for Cross-Encoder: [[query, doc1], [query, doc2], ...]
        # We assume 'description' or 'page_content' holds the text
        pairs = []
        valid_docs = []
        
        for doc in docs:
            text = doc.get("description") or doc.get("page_content") or str(doc)
            pairs.append([query, text])
            valid_docs.append(doc)
            
        if not pairs:
            return docs[:top_k]

        # Predict scores
        scores = self.model.predict(pairs)
        
        # Attach scores and sort
        scored_results = []
        for i, doc in enumerate(valid_docs):
            doc_copy = doc.copy()
            doc_copy["score"] = float(scores[i])
            scored_results.append(doc_copy)
            
        # Sort descending by score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_results[:top_k]

# Global instance
reranker = RerankerService()
