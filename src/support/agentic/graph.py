"""
LangGraph workflow for Agentic RAG: Router -> Retrieve -> Evaluate -> (optional) Web Fallback.

v2.0 change: build_agentic_graph() accepts use_llm_evaluate flag.
When True, swaps in llm_evaluate_node; when False (default), uses the original
rule-based evaluate_node — zero behaviour change for existing callers.

Thread safety: singleton rebuild is protected by _graph_lock.
"""
import threading
from typing import Optional

from langgraph.graph import StateGraph, START, END

from src.support.agentic.state import RAGState
from src.support.agentic.nodes import (
    router_node,
    retrieve_node,
    evaluate_node,
    llm_evaluate_node,
    web_fallback_node,
)
from src.infra.config import USE_LLM_EVALUATE
from src.infra.utils import setup_logger

logger = setup_logger(__name__)

_agentic_graph = None
_agentic_graph_llm_flag: Optional[bool] = None  # flag used when graph was built
_graph_lock = threading.Lock()


def _route_after_evaluate(state: RAGState):
    """Route to web_fallback if need_more else END."""
    if state.get("need_more") and state.get("retry_count", 0) < 1:
        return "web_fallback"
    return END


def build_agentic_graph(use_llm_evaluate: Optional[bool] = None):
    """
    Build and compile the Agentic RAG StateGraph.

    Args:
        use_llm_evaluate: Override the USE_LLM_EVALUATE config value.
                          None → read from config (default behaviour).
    """
    _use_llm: bool = (
        use_llm_evaluate if use_llm_evaluate is not None else USE_LLM_EVALUATE
    )

    builder = StateGraph(RAGState)
    builder.add_node("router", router_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("web_fallback", web_fallback_node)

    if _use_llm:
        builder.add_node("evaluate", llm_evaluate_node)
        logger.info("Agentic graph: evaluate=llm_evaluate_node (USE_LLM_EVALUATE=True)")
    else:
        builder.add_node("evaluate", evaluate_node)
        logger.info("Agentic graph: evaluate=rule_evaluate_node (USE_LLM_EVALUATE=False)")

    builder.add_edge(START, "router")
    builder.add_edge("router", "retrieve")
    builder.add_edge("retrieve", "evaluate")
    builder.add_conditional_edges("evaluate", _route_after_evaluate)
    builder.add_edge("web_fallback", END)

    graph = builder.compile()
    return graph


def get_agentic_graph(use_llm_evaluate: Optional[bool] = None):
    """
    Lazy-initialize and return the compiled Agentic graph.

    Rebuilds the graph if use_llm_evaluate differs from the cached flag,
    ensuring feature flag changes take effect without a server restart.
    Thread-safe via _graph_lock.
    """
    global _agentic_graph, _agentic_graph_llm_flag

    _requested = use_llm_evaluate if use_llm_evaluate is not None else USE_LLM_EVALUATE

    # Fast path: no lock needed if graph is already built with matching flag
    if _agentic_graph is not None and _agentic_graph_llm_flag == _requested:
        return _agentic_graph

    with _graph_lock:
        # Double-checked locking
        if _agentic_graph is None or _agentic_graph_llm_flag != _requested:
            _agentic_graph = build_agentic_graph(use_llm_evaluate=_requested)
            _agentic_graph_llm_flag = _requested

    return _agentic_graph


def reset_agentic_graph():
    """
    Force the singleton to be rebuilt on next call to get_agentic_graph().
    Useful for testing or runtime flag updates.
    """
    global _agentic_graph, _agentic_graph_llm_flag
    with _graph_lock:
        _agentic_graph = None
        _agentic_graph_llm_flag = None
    logger.info("Agentic graph singleton reset")
