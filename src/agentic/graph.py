"""
LangGraph workflow for Agentic RAG: Router -> Retrieve -> Evaluate -> (optional) Web Fallback.
"""
from langgraph.graph import StateGraph, START, END

from src.agentic.state import RAGState
from src.agentic.nodes import router_node, retrieve_node, evaluate_node, web_fallback_node
from src.utils import setup_logger

logger = setup_logger(__name__)

_agentic_graph = None


def _route_after_evaluate(state: RAGState):
    """Route to web_fallback if need_more else END."""
    if state.get("need_more") and state.get("retry_count", 0) < 1:
        return "web_fallback"
    return END


def build_agentic_graph():
    """Build and compile the Agentic RAG StateGraph."""
    builder = StateGraph(RAGState)

    builder.add_node("router", router_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("web_fallback", web_fallback_node)

    builder.add_edge(START, "router")
    builder.add_edge("router", "retrieve")
    builder.add_edge("retrieve", "evaluate")
    builder.add_conditional_edges("evaluate", _route_after_evaluate)
    builder.add_edge("web_fallback", END)

    graph = builder.compile()
    logger.info("Agentic RAG graph built and compiled")
    return graph


def get_agentic_graph():
    """Lazy-initialize and return the compiled Agentic graph."""
    global _agentic_graph
    if _agentic_graph is None:
        _agentic_graph = build_agentic_graph()
    return _agentic_graph
