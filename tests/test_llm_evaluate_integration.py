"""
Integration tests for llm_evaluate_node inside the full LangGraph workflow.

These tests exercise the graph compilation, node routing, and end-to-end state
transitions — things the unit tests deliberately skip by mocking at the node level.

Markers
-------
    pytest -m integration        # run only integration tests
    pytest -m "not integration"  # skip integration tests (fast CI)

Coverage targets
----------------
    ✓ Graph compiles with USE_LLM_EVALUATE=True and False
    ✓ Correct evaluate node is wired (llm vs rule)
    ✓ END routing when need_more=False (no web fallback)
    ✓ web_fallback routing when need_more=True
    ✓ retry_count=1 hard stop prevents infinite loop
    ✓ Cache hit skips LLM on second identical call
    ✓ Thread-safety: concurrent get_agentic_graph() rebuilds
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.support.agentic.graph import build_agentic_graph, get_agentic_graph, reset_agentic_graph
from src.support.agentic.nodes import EvaluationResult, llm_evaluate_node
from src.support.agentic.llm_evaluate_cache import llm_evaluate_cache


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_and_clear():
    """Ensure graph singleton and cache are clean for every test."""
    reset_agentic_graph()
    llm_evaluate_cache.clear()
    yield
    reset_agentic_graph()
    llm_evaluate_cache.clear()


def _mock_llm(score: float = 0.85, reason: str = "relevant"):
    """Return an LLM mock that responds with EvaluationResult via Path A."""
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=EvaluationResult(score=score, reason=reason)
    )
    return llm


def _batch_meta(isbn_list):
    return {isbn: {"title": f"Book {isbn}", "description": "test"} for isbn in isbn_list}


# ── 1. Graph compilation ──────────────────────────────────────────────────────

@pytest.mark.integration
def test_graph_compiles_rule_mode():
    """build_agentic_graph(use_llm_evaluate=False) should compile without error."""
    graph = build_agentic_graph(use_llm_evaluate=False)
    assert graph is not None
    assert "evaluate" in graph.nodes
    assert "router" in graph.nodes
    assert "retrieve" in graph.nodes
    assert "web_fallback" in graph.nodes


@pytest.mark.integration
def test_graph_compiles_llm_mode():
    """build_agentic_graph(use_llm_evaluate=True) should compile without error."""
    graph = build_agentic_graph(use_llm_evaluate=True)
    assert graph is not None
    assert "evaluate" in graph.nodes


# ── 2. End-to-end: high quality → END (no web_fallback) ──────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_high_quality_routes_to_end():
    """
    Full graph invocation with LLM evaluate:
    router → retrieve(mocked) → llm_evaluate(score=0.85) → END.
    web_fallback should NOT be called.
    """
    isbn_list = ["isbn_001", "isbn_002", "isbn_003"]
    web_fallback_called = []

    async def mock_web_fallback(state, config=None):
        web_fallback_called.append(True)
        return {"isbn_list": state.get("isbn_list", []), "need_more": False, "retry_count": 1}

    def mock_retrieve(state):
        return {"isbn_list": isbn_list}

    def mock_router(state):
        return {
            "strategy": "deep",
            "temporal": False,
            "freshness_fallback": False,
            "freshness_threshold": 3,
            "decision_reason": "mocked",
        }

    with patch("src.rag.llm.LLMFactory.create", return_value=_mock_llm(0.85)), \
         patch("src.data.stores.metadata_store.metadata_store") as ms, \
         patch("src.support.agentic.graph.router_node", side_effect=mock_router), \
         patch("src.support.agentic.graph.retrieve_node", side_effect=mock_retrieve), \
         patch("src.support.agentic.graph.web_fallback_node", side_effect=mock_web_fallback):

        ms.batch_get_book_metadata.return_value = _batch_meta(isbn_list)

        graph = build_agentic_graph(use_llm_evaluate=True)
        result = await graph.ainvoke(
            {"query": "books about loss", "retry_count": 0},
            config={"configurable": {"llm_provider": "mock"}},
        )

    assert result["need_more"] is False
    assert result["quality_score"] == pytest.approx(0.85)
    assert result["llm_evaluate_failed"] is False
    assert web_fallback_called == [], "web_fallback must NOT be triggered for high-quality results"


# ── 3. End-to-end: low quality → web_fallback triggered ──────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_low_quality_triggers_web_fallback():
    """
    score=0.3 < threshold=0.6 → need_more=True → web_fallback node executed.
    """
    isbn_list = ["isbn_001", "isbn_002", "isbn_003"]
    web_fallback_called = []

    async def mock_web_fallback(state, config=None):
        web_fallback_called.append(True)
        return {"isbn_list": state.get("isbn_list", []) + ["isbn_new"],
                "need_more": False, "retry_count": 1}

    def mock_retrieve(state):
        return {"isbn_list": isbn_list}

    def mock_router(state):
        return {"strategy": "deep", "temporal": False,
                "freshness_fallback": False, "freshness_threshold": 3,
                "decision_reason": "mocked"}

    with patch("src.rag.llm.LLMFactory.create", return_value=_mock_llm(0.3)), \
         patch("src.data.stores.metadata_store.metadata_store") as ms, \
         patch("src.support.agentic.graph.router_node", side_effect=mock_router), \
         patch("src.support.agentic.graph.retrieve_node", side_effect=mock_retrieve), \
         patch("src.support.agentic.graph.web_fallback_node", side_effect=mock_web_fallback):

        ms.batch_get_book_metadata.return_value = _batch_meta(isbn_list)

        graph = build_agentic_graph(use_llm_evaluate=True)
        result = await graph.ainvoke(
            {"query": "unrelated query", "retry_count": 0},
            config={"configurable": {"llm_provider": "mock"}},
        )

    assert web_fallback_called == [True], "web_fallback must be triggered once"
    assert result["retry_count"] == 1


# ── 4. retry_count=1 hard stop ───────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_count_1_prevents_infinite_loop():
    """
    When retry_count is already 1, evaluate must return need_more=False
    without calling LLM — preventing infinite graph loops.
    """
    with patch("src.rag.llm.LLMFactory.create") as factory, \
         patch("src.data.stores.metadata_store.metadata_store"):

        result = await llm_evaluate_node(
            {"query": "any", "isbn_list": ["a", "b", "c"],
             "retry_count": 1, "freshness_fallback": False, "freshness_threshold": 3},
            {"configurable": {"llm_provider": "mock"}},
        )

    factory.assert_not_called()
    assert result["need_more"] is False


# ── 5. Cache hit skips LLM ────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_hit_skips_llm():
    """Second call with identical (query, isbn_list) must use cache, not invoke LLM."""
    isbn_list = ["isbn_001", "isbn_002", "isbn_003"]
    call_count = [0]

    async def counting_ainvoke(*args, **kwargs):
        call_count[0] += 1
        return EvaluationResult(score=0.9, reason="cached")

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = counting_ainvoke

    state = {
        "query": "books about adventure",
        "isbn_list": isbn_list,
        "retry_count": 0,
        "freshness_fallback": False,
        "freshness_threshold": 3,
    }
    cfg = {"configurable": {"llm_provider": "mock"}}

    with patch("src.rag.llm.LLMFactory.create", return_value=mock_llm), \
         patch("src.data.stores.metadata_store.metadata_store") as ms:
        ms.batch_get_book_metadata.return_value = _batch_meta(isbn_list)

        first = await llm_evaluate_node(state, cfg)
        second = await llm_evaluate_node(state, cfg)

    assert call_count[0] == 1, f"LLM should be called exactly once; called {call_count[0]} times"
    assert first["quality_score"] == second["quality_score"]


# ── 6. Thread safety: concurrent singleton rebuild ────────────────────────────

@pytest.mark.integration
def test_singleton_thread_safety():
    """
    Concurrent calls to get_agentic_graph() from multiple threads must
    all return a valid, compiled graph without raising exceptions.
    """
    reset_agentic_graph()
    graphs = []
    errors = []

    def worker():
        try:
            g = get_agentic_graph(use_llm_evaluate=False)
            graphs.append(id(g))
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"
    assert len(graphs) == 8
    # All threads should have received the same singleton instance
    assert len(set(graphs)) == 1, "Expected single graph instance across all threads"


# ── 7. Cache stats observable ────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_stats_after_evaluation():
    """Cache stats reflect correct live entry count post-evaluation."""
    isbn_list = ["isbn_001", "isbn_002", "isbn_003"]
    mock_llm = _mock_llm(0.75)
    state = {
        "query": "unique stats test query",
        "isbn_list": isbn_list,
        "retry_count": 0,
        "freshness_fallback": False,
        "freshness_threshold": 3,
    }

    with patch("src.rag.llm.LLMFactory.create", return_value=mock_llm), \
         patch("src.data.stores.metadata_store.metadata_store") as ms:
        ms.batch_get_book_metadata.return_value = _batch_meta(isbn_list)
        await llm_evaluate_node(state, {"configurable": {"llm_provider": "mock"}})

    stats = llm_evaluate_cache.stats()
    assert stats["live_entries"] == 1
    assert stats["total_entries"] == 1


# ── 8. Cache TTL expiry ────────────────────────────────────────────────────────

@pytest.mark.integration
def test_cache_ttl_expiry():
    """Entry expires after TTL; subsequent get returns None (cache miss)."""
    import time
    cache = llm_evaluate_cache.__class__(ttl_seconds=0.05)  # 50ms TTL

    query = "ttl test query"
    isbn_list = ["isbn_001", "isbn_002"]
    max_books = 5
    value = {"need_more": False, "quality_score": 0.9,
             "quality_reason": "test", "llm_evaluate_failed": False}

    cache.set(query, isbn_list, max_books, value)
    assert cache.get(query, isbn_list, max_books) is not None, "Should be a cache hit"

    time.sleep(0.1)  # Wait for TTL to expire
    assert cache.get(query, isbn_list, max_books) is None, "Should be a cache miss after TTL"
    assert cache.stats()["live_entries"] == 0
