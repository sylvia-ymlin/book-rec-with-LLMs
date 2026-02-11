import re
from typing import Dict, Any, List
from src.utils import setup_logger

logger = setup_logger(__name__)

class QueryRouter:
    """
    Intelligent Router for the RAG Pipeline.
    Classifies user queries to select the optimal retrieval strategy.
    
    Strategies:
    1. EXACT (ISBN/ID) -> Pure BM25 (High Precision, No Rerank noise).
    2. FAST (Keywords) -> Hybrid (RRF), No Rerank (Low Latency).
    3. DEEP (Complex)  -> Hybrid + Rerank (High Latency, High contextual relevance).
    """
    
    def __init__(self):
        # Regex for ISBN-10 and ISBN-13
        self.isbn_pattern = re.compile(r'^(?:\d{9}[\dX]|\d{13})$')
    
    def route(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and return retrieval parameters.
        Returns dict with: 'strategy', 'hybrid_alpha', 'rerank'
        """
        cleaned_query = query.strip()
        words = cleaned_query.split()
        
        # 1. Check for ISBN (Exact Match)
        # Remove hyphens/spaces for check
        normalized = cleaned_query.replace("-", "").replace(" ", "")
        if self.isbn_pattern.match(normalized):
            logger.info(f"Router: Detected ISBN -> EXACT Strategy ({normalized})")
            return {"strategy": "exact", "alpha": 1.0, "rerank": False, "k_final": 5}

        # 2. Check for Temporal Keywords (Freshness Bias)
        temporal_keywords = {"new", "newest", "latest", "recent", "modern", "contemporary", "2020", "2021", "2022", "2023", "2024", "2025"}
        is_temporal = any(word.lower() in temporal_keywords for word in words)
        
        # 3. Check for Detail-Oriented Queries (Triggers Small-to-Big)
        # These are queries asking about specific plot points, reactions, or hidden details
        detail_keywords = {"twist", "ending", "spoiler", "readers", "felt", "cried", "hated", "loved", 
                          "review", "opinion", "think", "unreliable", "narrator", "realize", "find out"}
        is_detail = any(word.lower() in detail_keywords for word in words)
        
        if is_detail:
            logger.info(f"Router: Detected Detail Query -> SMALL_TO_BIG Strategy")
            return {
                "strategy": "small_to_big",
                "rerank": False,  # Small-to-Big already does precision matching
                "k_final": 5,
                "temporal": is_temporal
            }
        
        # 4. Check for Simple Keyword Search (Short queries)
        if len(words) <= 2:
             logger.info(f"Router: Detected Keyword -> FAST Strategy (Temporal={is_temporal})")
             return {
                 "strategy": "fast",
                 "rerank": False,     # Skip expensive rerank
                 "k_final": 5,
                 "temporal": is_temporal
             }

        # 5. Default to Deep Search
        logger.info(f"Router: Detected Natural Language -> DEEP Strategy (Temporal={is_temporal})")
        return {
            "strategy": "deep",
            "rerank": True,
            "k_final": 10,
            "temporal": is_temporal
        }

