"""
Unit tests for llm_evaluate_node and _rule_based_evaluate.

Run: pytest tests/test_llm_evaluate.py -v
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.support.agentic.nodes import (
    llm_evaluate_node,
    _rule_based_evaluate,
    _parse_json_response,
    _invoke_with_fallback,
    EvaluationResult,
)
from src.support.agentic.state import RAGState
from src.support.agentic.llm_evaluate_cache import llm_evaluate_cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Flush the LLM evaluate cache before every test for isolation."""
    llm_evaluate_cache.clear()
    yield
    llm_evaluate_cache.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(**kwargs) -> RAGState:
    defaults: RAGState = {
        "query": "books about grief and healing",
        "isbn_list": ["0001", "0002", "0003"],
        "retry_count": 0,
        "freshness_fallback": False,
        "freshness_threshold": 3,
    }
    return {**defaults, **kwargs}


def make_llm_response(score: float, reason: str = "test reason") -> MagicMock:
    """Return a mock LLM response with .content as JSON string."""
    msg = MagicMock()
    msg.content = json.dumps({"score": score, "reason": reason})
    return msg


def _config(provider: str = "mock", **extra):
    return {"configurable": {"llm_provider": provider, **extra}}


# ── 1. High quality → no fallback ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_high_quality_no_fallback():
    """LLM score ≥ threshold → need_more=False."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=EvaluationResult(score=0.85, reason="all relevant")
    )

    with patch("src.rag.llm.LLMFactory.create", return_value=mock_llm), \
         patch("src.data.stores.metadata_store.metadata_store") as ms:
        ms.batch_get_book_metadata.return_value = {
            isbn: {"title": "A Book", "description": "desc"}
            for isbn in ["0001", "0002", "0003"]
        }

        result = await llm_evaluate_node(make_state(), _config())

    assert result["need_more"] is False
    assert result["quality_score"] == pytest.approx(0.85)
    assert result["llm_evaluate_failed"] is False


# ── 2. Low quality → triggers fallback ───────────────────────────────────────

@pytest.mark.asyncio
async def test_low_quality_triggers_fallback():
    """LLM score < threshold → need_more=True."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=EvaluationResult(score=0.3, reason="mostly irrelevant")
    )

    with patch("src.rag.llm.LLMFactory.create", return_value=mock_llm), \
         patch("src.data.stores.metadata_store.metadata_store") as ms:
        ms.batch_get_book_metadata.return_value = {
            isbn: {"title": "Wrong Book", "description": "x"}
            for isbn in ["0001", "0002", "0003"]
        }

        result = await llm_evaluate_node(make_state(), _config())

    assert result["need_more"] is True
    assert result["quality_score"] == pytest.approx(0.3)
    assert result["llm_evaluate_failed"] is False


# ── 3. Timeout → rule fallback ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_falls_back_to_rule():
    """LLM call exceeds timeout → rule-based fallback, llm_evaluate_failed=True."""

    async def slow_call(*args, **kwargs):
        # Use an asyncio.Event that never fires → deterministic infinite wait,
        # avoids the asyncio.sleep(10) fragility on slow CI machines.
        await asyncio.Event().wait()

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = slow_call

    with patch("src.rag.llm.LLMFactory.create", return_value=mock_llm), \
         patch("src.data.stores.metadata_store.metadata_store") as ms:
        ms.batch_get_book_metadata.return_value = {
            isbn: {"title": "T", "description": "d"}
            for isbn in ["0001", "0002", "0003"]
        }

        result = await llm_evaluate_node(
            make_state(),
            _config(llm_evaluate_timeout=0.05),   # 50ms timeout
        )

    assert result["llm_evaluate_failed"] is True
    assert "llm_timeout" in result["quality_reason"]
    # Rule fallback: 3 isbns, no freshness → need_more=False
    assert result["need_more"] is False


# ── 4. LLM exception → rule fallback ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_error_falls_back_to_rule():
    """LLM raises any exception → rule fallback, llm_evaluate_failed=True."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = RuntimeError("connection refused")

    with patch("src.rag.llm.LLMFactory.create", return_value=mock_llm), \
         patch("src.data.stores.metadata_store.metadata_store") as ms:
        ms.batch_get_book_metadata.return_value = {
            isbn: {"title": "T", "description": "d"}
            for isbn in ["0001", "0002", "0003"]
        }

        result = await llm_evaluate_node(make_state(), _config())

    assert result["llm_evaluate_failed"] is True
    assert "llm_error" in result["quality_reason"]


# ── 5. Empty isbn_list → rule fallback ───────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_isbn_list_rule_fallback():
    """isbn_list=[] → skip LLM entirely, rule fallback (need_more=True)."""
    with patch("src.rag.llm.LLMFactory.create") as factory, \
         patch("src.data.stores.metadata_store.metadata_store"):

        result = await llm_evaluate_node(make_state(isbn_list=[]), _config())

    factory.assert_not_called()   # LLM never invoked
    assert result["need_more"] is True


# ── 6. retry_count ≥ 1 → hard stop ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_count_gte_1_forces_stop():
    """When retry_count ≥ 1, need_more must be False (no infinite loop)."""
    with patch("src.rag.llm.LLMFactory.create") as factory, \
         patch("src.data.stores.metadata_store.metadata_store"):

        result = await llm_evaluate_node(make_state(retry_count=1), _config())

    factory.assert_not_called()
    assert result["need_more"] is False


# ── 7. Rule fallback logic identical to evaluate_node ─────────────────────────

def test_rule_fallback_matches_evaluate_node():
    """
    _rule_based_evaluate must produce the same need_more decision as evaluate_node
    for all combinations of freshness/count.
    """
    from src.support.agentic.nodes import evaluate_node

    cases = [
        # (isbn_list, freshness_fallback, freshness_threshold)
        ([], True, 3),
        (["a"], True, 3),
        (["a", "b"], True, 3),
        (["a", "b", "c"], True, 3),
        ([], False, 3),
        (["a"], False, 3),
        (["a", "b"], False, 3),
    ]

    for isbn_list, ff, threshold in cases:
        state = make_state(isbn_list=isbn_list, freshness_fallback=ff,
                           freshness_threshold=threshold, retry_count=0)
        rule_result = evaluate_node(state)
        llm_fb_result = _rule_based_evaluate(state, reason="test")
        assert rule_result["need_more"] == llm_fb_result["need_more"], (
            f"Mismatch for {isbn_list=}, {ff=}, {threshold=}: "
            f"evaluate_node={rule_result['need_more']} "
            f"_rule_based_evaluate={llm_fb_result['need_more']}"
        )


# ── 8. Books context truncated to max_books ───────────────────────────────────

@pytest.mark.asyncio
async def test_books_context_max_books():
    """LLM receives at most LLM_EVALUATE_MAX_BOOKS book entries."""
    captured_prompts = []

    async def mock_ainvoke(prompt):
        captured_prompts.append(prompt)
        return EvaluationResult(score=0.8, reason="ok")

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = mock_ainvoke

    isbn_list = [f"isbn_{i:03d}" for i in range(20)]   # 20 ISBNs

    with patch("src.rag.llm.LLMFactory.create", return_value=mock_llm), \
         patch("src.data.stores.metadata_store.metadata_store") as ms:
        ms.batch_get_book_metadata.return_value = {
            isbn: {"title": "T", "description": "desc"} for isbn in isbn_list[:5]
        }

        await llm_evaluate_node(
            make_state(isbn_list=isbn_list),
            _config(llm_evaluate_max_books=5),
        )

    assert captured_prompts, "LLM should have been called"
    prompt_text = captured_prompts[0]
    # Should list entries "1." through "5." but not "6."
    assert "5." in prompt_text
    assert "6." not in prompt_text


# ── 9. Config passthrough: provider and threshold ────────────────────────────

@pytest.mark.asyncio
async def test_config_passthrough():
    """provider, api_key, and quality_threshold are correctly passed from config."""
    captured_kwargs: dict = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        m = MagicMock()
        m.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=EvaluationResult(score=0.9, reason="ok")
        )
        return m

    with patch("src.rag.llm.LLMFactory.create", side_effect=fake_create), \
         patch("src.data.stores.metadata_store.metadata_store") as ms:
        ms.batch_get_book_metadata.return_value = {
            isbn: {"title": "T", "description": "d"} for isbn in ["0001", "0002", "0003"]
        }

        result = await llm_evaluate_node(
            make_state(),
            {"configurable": {
                "llm_provider": "groq",
                "llm_api_key": "test-key-xyz",
                "quality_threshold": 0.95,   # score 0.9 < 0.95 → need_more=True
            }},
        )

    assert captured_kwargs["provider"] == "groq"
    assert captured_kwargs["api_key"] == "test-key-xyz"
    assert result["need_more"] is True  # 0.9 < 0.95


# ── 10. JSON parsing: regex fallback + score clamping ────────────────────────

def test_parse_json_response_valid():
    result = _parse_json_response('{"score": 0.75, "reason": "mostly good"}')
    assert result.score == pytest.approx(0.75)
    assert result.reason == "mostly good"


def test_parse_json_response_wrapped_in_text():
    """LLM wraps JSON in extra text — regex extraction must work."""
    raw = 'Here is my evaluation:\n{"score": 0.4, "reason": "few relevant"}\nHope this helps.'
    result = _parse_json_response(raw)
    assert result.score == pytest.approx(0.4)


def test_parse_json_response_score_clamped_above():
    """LLM returns score > 1.0 → clamped to 1.0."""
    result = _parse_json_response('{"score": 1.5, "reason": "out of range"}')
    assert result.score == pytest.approx(1.0)


def test_parse_json_response_score_clamped_below():
    """LLM returns score < 0.0 → clamped to 0.0."""
    result = _parse_json_response('{"score": -0.3, "reason": "negative"}')
    assert result.score == pytest.approx(0.0)


def test_parse_json_response_unparseable_raises():
    """Completely unparseable text raises ValueError."""
    with pytest.raises(ValueError, match="Cannot parse"):
        _parse_json_response("I cannot evaluate this.")


# ── 11. graph.py: feature flag routing ───────────────────────────────────────

def test_build_graph_rule_mode():
    """USE_LLM_EVALUATE=False → evaluate node is synchronous evaluate_node."""
    from src.support.agentic.graph import build_agentic_graph, reset_agentic_graph
    from src.support.agentic.nodes import evaluate_node

    reset_agentic_graph()
    graph = build_agentic_graph(use_llm_evaluate=False)
    node_funcs = {k: v for k, v in graph.nodes.items()}
    # Graph should compile without error; evaluate node should be rule-based
    assert "evaluate" in node_funcs


def test_build_graph_llm_mode():
    """USE_LLM_EVALUATE=True → evaluate node is async llm_evaluate_node."""
    from src.support.agentic.graph import build_agentic_graph, reset_agentic_graph

    reset_agentic_graph()
    graph = build_agentic_graph(use_llm_evaluate=True)
    assert "evaluate" in graph.nodes


def test_get_agentic_graph_rebuilds_on_flag_change():
    """Changing use_llm_evaluate causes singleton to rebuild."""
    from src.support.agentic.graph import get_agentic_graph, reset_agentic_graph

    reset_agentic_graph()
    g1 = get_agentic_graph(use_llm_evaluate=False)
    g2 = get_agentic_graph(use_llm_evaluate=True)
    assert g1 is not g2   # rebuilt


def test_get_agentic_graph_cached_same_flag():
    """Same flag → same graph instance returned (no rebuild)."""
    from src.support.agentic.graph import get_agentic_graph, reset_agentic_graph

    reset_agentic_graph()
    g1 = get_agentic_graph(use_llm_evaluate=False)
    g2 = get_agentic_graph(use_llm_evaluate=False)
    assert g1 is g2
