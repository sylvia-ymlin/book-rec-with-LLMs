"""
Agentic RAG workflow powered by LangGraph.

Provides a stateful retrieval pipeline: Router -> Retrieve -> Evaluate -> (optional) Web Fallback.
Enables LLM-based evaluation of result quality and conditional web search when local results
are insufficient.
"""
from src.support.agentic.graph import build_agentic_graph, get_agentic_graph

__all__ = ["build_agentic_graph", "get_agentic_graph"]
