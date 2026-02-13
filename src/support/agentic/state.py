"""
State schema for the Agentic RAG LangGraph workflow.
"""
from typing import TypedDict, Optional


class RAGState(TypedDict, total=False):
    """State passed through the Agentic RAG graph."""

    query: str
    category: str
    strategy: str
    temporal: bool
    freshness_fallback: bool
    freshness_threshold: int
    isbn_list: list[str]
    need_more: bool
    retry_count: int
    decision_reason: str
