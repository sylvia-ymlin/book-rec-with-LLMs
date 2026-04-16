"""
State schema for the Agentic RAG LangGraph workflow.
"""
from typing import TypedDict, Optional


class RAGState(TypedDict, total=False):
    """State passed through the Agentic RAG graph."""

    # ── Core retrieval state ──────────────────────────────────────────────────
    query: str
    category: str
    strategy: str               # exact | fast | deep | small_to_big
    temporal: bool
    freshness_fallback: bool
    freshness_threshold: int
    isbn_list: list[str]
    need_more: bool
    retry_count: int
    decision_reason: str

    # ── v2.0: LLM evaluate node ───────────────────────────────────────────────
    # quality_score: None → LLM not used (rule-based path or disabled)
    quality_score: Optional[float]
    # quality_reason: evaluation rationale or fallback reason code
    quality_reason: Optional[str]
    # llm_evaluate_failed: True when LLM timed out or errored → rule fallback used
    llm_evaluate_failed: bool

    # ── v2.1: observability ───────────────────────────────────────────────────
    # query_id: caller-provided trace ID; auto-generated (uuid4 prefix) if absent
    query_id: Optional[str]
