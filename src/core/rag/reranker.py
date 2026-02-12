"""
Reranker: Cross-Encoder (torch/ONNX) or ColBERT (optional).
Backend selectable via RERANKER_BACKEND env: cross_encoder | onnx | colbert.
ONNX ~2x faster than torch; ColBERT requires llama-index-postprocessor-colbert-rerank.
"""
from typing import List, Any

from src.config import RERANKER_BACKEND
from src.utils import setup_logger


logger = setup_logger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_cross_encoder(backend: str):
    """Load CrossEncoder with torch or ONNX backend. Falls back to torch if ONNX fails."""
    from sentence_transformers import CrossEncoder
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    be = "onnx" if backend == "onnx" else "torch"

    try:
        logger.info(
            "Loading Reranker (%s) backend=%s on %s...",
            DEFAULT_RERANKER_MODEL,
            be,
            device,
        )
        model = CrossEncoder(DEFAULT_RERANKER_MODEL, device=device, backend=be)
        logger.info("Reranker model loaded.")
        return model
    except Exception as e:
        if be == "onnx":
            logger.warning(
                "ONNX backend failed (pip install onnxruntime?), falling back to torch: %s",
                e,
            )
            return CrossEncoder(
                DEFAULT_RERANKER_MODEL, device=device, backend="torch"
            )
        raise


def _load_colbert():
    """Load ColBERT reranker via llama-index (optional dep)."""
    try:
        from llama_index.postprocessor.colbert_rerank import ColbertRerank

        return ColbertRerank(
            model_name="colbert-ir/colbertv2.0",
            top_n=10,
        )
    except ImportError as e:
        logger.warning(
            "ColBERT not available (pip install llama-index-postprocessor-colbert-rerank): %s",
            e,
        )
        return None


def _get_text(doc: Any) -> str:
    if hasattr(doc, "page_content"):
        return doc.page_content
    return doc.get("description") or doc.get("page_content") or str(doc)


def _set_score(doc: Any, score: float) -> None:
    if hasattr(doc, "metadata"):
        doc.metadata["relevance_score"] = score
    else:
        doc["score"] = score


def _get_score(doc: Any) -> float:
    if hasattr(doc, "metadata"):
        return doc.metadata.get("relevance_score", 0)
    return doc.get("score", 0)


class RerankerService:
    """
    Singleton reranker: Cross-Encoder (torch/ONNX) or ColBERT.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance._backend = None
        return cls._instance

    def __init__(self):
        if self.model is None:
            self._load_model()

    def _load_model(self):
        backend = (RERANKER_BACKEND or "").lower()

        if backend == "colbert":
            self.model = _load_colbert()
            self._backend = "colbert" if self.model else "cross_encoder"
            if self._backend == "cross_encoder":
                self.model = _load_cross_encoder("torch")
        else:
            self._backend = "onnx" if backend == "onnx" else "cross_encoder"
            self.model = _load_cross_encoder(self._backend)

    def rerank(self, query: str, docs: List[Any], top_k: int = 5) -> List[Any]:
        """
        Rerank documents by relevance to query.
        docs: List of dicts or LangChain Document with description/page_content.
        """
        if not self.model or not docs:
            return docs[:top_k]

        if self._backend == "colbert":
            return self._rerank_colbert(query, docs, top_k)
        return self._rerank_cross_encoder(query, docs, top_k)

    def _rerank_cross_encoder(
        self, query: str, docs: List[Any], top_k: int
    ) -> List[Any]:
        pairs = [[query, _get_text(d)] for d in docs]
        scores = self.model.predict(pairs)

        for i, doc in enumerate(docs):
            _set_score(doc, float(scores[i]))

        docs.sort(key=_get_score, reverse=True)
        return docs[:top_k]

    def _rerank_colbert(
        self, query: str, docs: List[Any], top_k: int
    ) -> List[Any]:
        from llama_index.schema import NodeWithScore, TextNode

        # Keep ref to original doc for metadata (isbn, etc.)
        nodes = []
        for d in docs:
            node = TextNode(text=_get_text(d), metadata={"__original": d})
            nodes.append(NodeWithScore(node=node, score=0.0))

        reranked = self.model.postprocess_nodes(nodes, query_str=query)

        result = []
        for nws in reranked[:top_k]:
            orig = getattr(nws.node, "metadata", {}).get("__original")
            if orig is not None:
                _set_score(orig, float(nws.score or 0))
                result.append(orig)
            else:
                from langchain_core.documents import Document

                doc = Document(
                    page_content=nws.node.text,
                    metadata={"relevance_score": float(nws.score or 0)},
                )
                result.append(doc)
        return result


reranker = RerankerService()

__all__ = ["RerankerService", "reranker"]

