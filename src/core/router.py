import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils import setup_logger

logger = setup_logger(__name__)


class QueryRouter:
    """
    Intelligent Router for the RAG Pipeline.
    Classifies user queries to select the optimal retrieval strategy.

    Uses model-based intent classifier when available; falls back to rule-based
    heuristics when classifier not trained/loaded.

    Strategies:
    1. EXACT (ISBN/ID) -> Pure BM25 (High Precision, No Rerank noise).
    2. FAST (Keywords) -> Hybrid (RRF), No Rerank (Low Latency).
    3. DEEP (Complex)  -> Hybrid + Rerank (High Latency, High contextual relevance).
    
    Freshness-Aware Routing:
    - Detects queries asking for "new", "latest", or specific years (2024, 2025, etc.)
    - Sets freshness_fallback=True to enable Web Search when local results insufficient
    
    Keywords loaded from config/router.json; overridable via ROUTER_DETAIL_KEYWORDS env.
    """

    def __init__(self, model_dir: str | Path | None = None):
        self.isbn_pattern = re.compile(r"^(?:\d{9}[\dX]|\d{13})$")
        if model_dir is None:
            from src.config import DATA_DIR
            model_dir = DATA_DIR / "model"
        self.model_dir = Path(model_dir)
        self._classifier = None

    def _get_classifier(self):
        """Lazy-load intent classifier when first needed."""
        if self._classifier is not None:
            return self._classifier
        try:
            from src.core.intent_classifier import IntentClassifier
            clf = IntentClassifier(self.model_dir / "intent_classifier.pkl")
            if clf.load():
                self._classifier = clf
        except Exception as e:
            logger.debug("Intent classifier not available: %s", e)
        return self._classifier

    def _detect_freshness(self, words: list) -> tuple[bool, bool, Optional[int]]:
        """
        Detect if query requires fresh content.
        
        Returns:
            (is_temporal, freshness_fallback, target_year)
            - is_temporal: Should apply temporal boost to local results
            - freshness_fallback: Should enable Web Search if local results insufficient  
            - target_year: Specific year user is looking for (if detected)
        """
        from datetime import datetime
        from src.config import ROUTER_FRESHNESS_KEYWORDS, ROUTER_STRONG_FRESHNESS_KEYWORDS

        current_year = datetime.now().year
        lower_words = {w.lower() for w in words}

        is_temporal = bool(lower_words & ROUTER_FRESHNESS_KEYWORDS)
        freshness_fallback = bool(lower_words & ROUTER_STRONG_FRESHNESS_KEYWORDS)
        
        # Extract explicit year from query
        target_year = None
        for word in words:
            if word.isdigit() and len(word) == 4:
                year = int(word)
                if 2000 <= year <= 2050:
                    target_year = year
                    # Recent years (within last 3 years) trigger freshness
                    if year >= current_year - 2:
                        is_temporal = True
                        freshness_fallback = True
                    break
        
        return is_temporal, freshness_fallback, target_year

    def _route_by_rules(
        self, 
        cleaned_query: str, 
        words: list, 
        is_temporal: bool,
        freshness_fallback: bool = False,
        target_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fallback: rule-based routing when classifier not loaded.
        Uses NL keywords (like, similar, recommend...) instead of brittle word-count.
        Book titles (e.g. "War and Peace", "The Lord of the Rings") -> FAST.
        """
        from src.config import ROUTER_DETAIL_KEYWORDS, ROUTER_NL_KEYWORDS

        base_result = {
            "temporal": is_temporal,
            "freshness_fallback": freshness_fallback,
            "freshness_threshold": 3,  # Trigger web search if < 3 results
            "target_year": target_year,
        }
        
        if any(w.lower() in ROUTER_DETAIL_KEYWORDS for w in words):
            logger.info("Router (rules): Detail Query -> SMALL_TO_BIG")
            return {**base_result, "strategy": "small_to_big", "alpha": 0.5, "rerank": False, "k_final": 5}
        # NL keywords indicate recommendation intent -> DEEP
        if any(w.lower() in ROUTER_NL_KEYWORDS for w in words):
            logger.info("Router (rules): NL keywords -> DEEP (Temporal=%s, Freshness=%s)", is_temporal, freshness_fallback)
            return {**base_result, "strategy": "deep", "alpha": 0.5, "rerank": True, "k_final": 10}
        # Short query without NL keywords: book title or keyword -> FAST
        if len(words) <= 6:
            logger.info("Router (rules): Keyword/Title -> FAST (Temporal=%s, Freshness=%s)", is_temporal, freshness_fallback)
            return {**base_result, "strategy": "fast", "alpha": 0.5, "rerank": False, "k_final": 5}
        logger.info("Router (rules): Long query -> DEEP (Temporal=%s, Freshness=%s)", is_temporal, freshness_fallback)
        return {**base_result, "strategy": "deep", "alpha": 0.5, "rerank": True, "k_final": 10}

    def route(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and return retrieval parameters.
        
        Returns dict with:
            - 'strategy': 'exact' | 'fast' | 'deep' | 'small_to_big'
            - 'alpha': float (hybrid search weight)
            - 'rerank': bool (use cross-encoder reranking)
            - 'k_final': int (number of results)
            - 'temporal': bool (apply temporal boost)
            - 'freshness_fallback': bool (enable web search if local results insufficient)
            - 'freshness_threshold': int (min local results before triggering web search)
            - 'target_year': int | None (specific year user requested)
        """
        cleaned_query = query.strip()
        words = cleaned_query.split()

        # 1. ISBN: keep regex (deterministic, correct)
        normalized = cleaned_query.replace("-", "").replace(" ", "")
        if self.isbn_pattern.match(normalized):
            logger.info("Router: ISBN -> EXACT (%s)", normalized)
            return {
                "strategy": "exact", 
                "alpha": 1.0, 
                "rerank": False, 
                "k_final": 5,
                "temporal": False,
                "freshness_fallback": False,
                "freshness_threshold": 1,
                "target_year": None,
            }

        # 2. Freshness detection (temporal boost + web fallback)
        is_temporal, freshness_fallback, target_year = self._detect_freshness(words)

        # 3. Model-based vs rule-based intent
        clf = self._get_classifier()
        if clf is not None:
            try:
                intent = clf.predict(cleaned_query)
                logger.info("Router (model): %s -> %s (Freshness=%s)", intent, intent.upper(), freshness_fallback)
                return {
                    "strategy": intent,
                    "alpha": 0.5,
                    "rerank": intent == "deep",
                    "k_final": 10 if intent == "deep" else 5,
                    "temporal": is_temporal,
                    "freshness_fallback": freshness_fallback,
                    "freshness_threshold": 3,
                    "target_year": target_year,
                }
            except Exception as e:
                logger.warning("Intent classifier failed, falling back to rules: %s", e)

        return self._route_by_rules(cleaned_query, words, is_temporal, freshness_fallback, target_year)

