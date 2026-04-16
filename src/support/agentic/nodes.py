"""
LangGraph nodes for the Agentic RAG workflow.

v2.1 changes (full feature release):
  - llm_evaluate_node: batch metadata fetch (no N+1), TTL cache, targeted
    retry on transient API errors, structured evaluation log, evaluation_id.
  - _invoke_with_fallback: Path A only catches API-level errors, not bare
    Exception; debug log on Path A failure for root-cause tracing.
  - _rule_based_evaluate: unchanged (used as degradation target).
"""
import asyncio
import json
import re
import time
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from src.support.agentic.state import RAGState
from src.infra.config import (
    TOP_K_INITIAL,
    LLM_EVALUATE_TIMEOUT,
    LLM_QUALITY_THRESHOLD,
    LLM_EVALUATE_MAX_BOOKS,
    LLM_EVALUATE_DESC_CHARS,
    LLM_PROVIDER,
    DEEPSEEK_API_KEY,
)
from src.rag.isbn_extractor import extract_isbn
from src.infra.utils import setup_logger
from src.support.agentic.llm_evaluate_cache import llm_evaluate_cache

logger = setup_logger(__name__)

# Transient HTTP status codes worth retrying (rate limit, server error)
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# ── Prometheus metrics ────────────────────────────────────────────────────────
# Registered at module load time so each metric is created exactly once.
# If prometheus_client is not installed, all three are None and callers
# guard with `if _metric is not None` before using.
try:
    from prometheus_client import Counter as _Counter, Histogram as _Histogram

    _EVAL_LATENCY = _Histogram(
        "llm_evaluate_latency_seconds",
        "LLM evaluate node latency in seconds",
        ["provider", "result"],
        buckets=[0.1, 0.25, 0.5, 1, 2, 4, 6, 10],
    )
    _EVAL_DEGRADATION = _Counter(
        "llm_evaluate_degradation_total",
        "LLM evaluate degraded to rule-based fallback",
        ["reason"],
    )
    _EVAL_CACHE_HITS = _Counter(
        "llm_evaluate_cache_hits_total",
        "LLM evaluate cache hits",
    )
except Exception:  # prometheus_client not installed or already registered
    _EVAL_LATENCY = None
    _EVAL_DEGRADATION = None
    _EVAL_CACHE_HITS = None


def _get_prom_metrics():
    """Return the three LLM-evaluate Prometheus metrics (or Nones if unavailable)."""
    return _EVAL_LATENCY, _EVAL_DEGRADATION, _EVAL_CACHE_HITS


# ── Structured output schema ──────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """LLM quality evaluation result. score is clamped to [0.0, 1.0]."""
    score: float = Field(ge=0.0, le=1.0, description="Semantic relevance score")
    reason: str = Field(max_length=300, description="One-sentence evaluation rationale")


# ── Prompt template (English instructions for cross-lingual stability) ────────

_EVALUATE_PROMPT = """\
You are a retrieval quality evaluator for a book recommendation system.

User query: {query}

Retrieved books:
{books_context}

Evaluate whether the retrieved books are semantically relevant to the user query.

Scoring rubric:
- 1.0  All books are highly relevant and satisfy the query intent
- 0.8  Most books are relevant
- 0.6  About half are relevant (borderline acceptable)
- 0.4  Only a few books are relevant, quality is poor
- 0.0  Books are completely unrelated to the query

Respond ONLY with valid JSON (no extra text):
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""


# ── Existing rule-based node (unchanged) ─────────────────────────────────────

async def router_node(state: RAGState) -> Dict[str, Any]:
    """Determine retrieval strategy using QueryRouter."""
    from src.rag.router import QueryRouter

    router = QueryRouter()
    # Execute routing in a thread to avoid blocking the event loop
    decision = await asyncio.to_thread(router.route, state["query"])
    logger.info(f"Agentic Router: {decision}")

    return {
        "strategy": decision["strategy"],
        "temporal": decision.get("temporal", False),
        "freshness_fallback": decision.get("freshness_fallback", False),
        "freshness_threshold": decision.get("freshness_threshold", 3),
        "decision_reason": f"routed to {decision['strategy']}",
    }


async def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """Execute retrieval based on strategy."""
    from src.rag.vector_db import VectorDB

    vector_db = VectorDB()
    strategy = state.get("strategy", "deep")
    query = state["query"]
    temporal = state.get("temporal", False)

    # Execute search in a thread to avoid blocking the event loop
    if strategy == "small_to_big":
        recs = await asyncio.to_thread(vector_db.small_to_big_search, query, k=TOP_K_INITIAL)
    elif strategy == "exact":
        recs = await asyncio.to_thread(
            vector_db.hybrid_search,
            query,
            k=TOP_K_INITIAL,
            alpha=1.0,
            rerank=False,
            temporal=False,
        )
    else:
        recs = await asyncio.to_thread(
            vector_db.hybrid_search,
            query,
            k=TOP_K_INITIAL,
            alpha=0.5,
            rerank=(strategy == "deep"),
            temporal=temporal,
        )

    isbn_list = []
    for doc in recs:
        isbn = extract_isbn(doc)
        if isbn:
            isbn_list.append(isbn)

    logger.info(f"Agentic Retrieve: {len(isbn_list)} results for strategy={strategy}")
    return {"isbn_list": isbn_list}


async def evaluate_node(state: RAGState) -> Dict[str, Any]:
    """
    Evaluate if local results are sufficient (rule-based).
    Triggers web fallback when: few results + freshness query, or very few results.
    """
    n_results = len(state.get("isbn_list", []))
    freshness_fallback = state.get("freshness_fallback", False)
    threshold = state.get("freshness_threshold", 3)
    retry_count = state.get("retry_count", 0)

    if retry_count >= 1:
        return {"need_more": False}
    if n_results == 0 and freshness_fallback:
        return {"need_more": True}
    if n_results < threshold and freshness_fallback:
        return {"need_more": True}
    if n_results < 2:
        return {"need_more": True}
    return {"need_more": False}


# ── v2.0: LLM semantic evaluate node ─────────────────────────────────────────

async def llm_evaluate_node(state: RAGState, config=None) -> Dict[str, Any]:
    """
    LLM semantic evaluate node (v2.0).

    Replaces rule-based evaluate_node when USE_LLM_EVALUATE=True.
    Assesses whether retrieved books are semantically relevant to the query,
    not just numerically sufficient.

    Degradation chain (in order):
        1. with_structured_output (OpenAI / Groq / modern Ollama)
        2. raw ainvoke + JSON regex parse  (older Ollama / any provider)
        3. rule-based fallback             (timeout / any unhandled error)

    Explicit dependencies:
        - metadata_store : SQLite book metadata (title, description)
        - LLMFactory     : LLM instance creation

    Config keys (via LangGraph configurable dict):
        llm_provider          : str   default "ollama"
        llm_api_key           : str   default None
        quality_threshold     : float default LLM_QUALITY_THRESHOLD (0.6)
        llm_evaluate_timeout  : float default LLM_EVALUATE_TIMEOUT (1.5)
    """
    from src.data.stores.metadata_store import metadata_store
    from src.rag.llm import LLMFactory

    # ── 1. Parse config ───────────────────────────────────────────────────────
    cfg: dict = {}
    if isinstance(config, dict):
        cfg = config.get("configurable", {}) or {}
    elif hasattr(config, "configurable"):
        cfg = getattr(config, "configurable", {}) or {}

    provider: str = cfg.get("llm_provider", LLM_PROVIDER)
    api_key: Optional[str] = cfg.get("llm_api_key", DEEPSEEK_API_KEY)
    threshold: float = float(cfg.get("quality_threshold", LLM_QUALITY_THRESHOLD))
    timeout_s: float = float(cfg.get("llm_evaluate_timeout", LLM_EVALUATE_TIMEOUT))
    max_books: int = int(cfg.get("llm_evaluate_max_books", LLM_EVALUATE_MAX_BOOKS))
    desc_chars: int = int(cfg.get("llm_evaluate_desc_chars", LLM_EVALUATE_DESC_CHARS))

    # ── 2. Pre-conditions ─────────────────────────────────────────────────────
    isbn_list: list = state.get("isbn_list", [])
    retry_count: int = state.get("retry_count", 0)
    query: str = state["query"]
    evaluation_id: str = state.get("query_id") or str(uuid.uuid4())[:8]

    if retry_count >= 1:
        return {
            "need_more": False,
            "quality_score": None,
            "quality_reason": "skip:retry_limit",
            "llm_evaluate_failed": False,
        }

    if len(isbn_list) < 2:
        logger.info("LLM Evaluate [%s]: insufficient results (%d), rule fallback",
                    evaluation_id, len(isbn_list))
        return _rule_based_evaluate(state, reason="rule:insufficient_count")

    # ── 3. Cache check ────────────────────────────────────────────────────────
    cached = llm_evaluate_cache.get(query, isbn_list, max_books)
    if cached is not None:
        logger.info("LLM Evaluate [%s]: cache hit, score=%.2f", evaluation_id,
                    cached.get("quality_score", 0))
        _hist, _deg, _cache = _get_prom_metrics()
        if _cache is not None:
            _cache.inc()
        return cached

    # ── 4. Build books context — batch metadata fetch (no N+1) ───────────────
    batch: dict = metadata_store.batch_get_book_metadata(isbn_list[:max_books])
    books_lines: list[str] = []
    for i, isbn in enumerate(isbn_list[:max_books]):
        meta: dict = batch.get(str(isbn).strip(), {})
        title: str = meta.get("title") or f"ISBN:{isbn}"
        desc: str = (meta.get("description") or "")[:desc_chars]
        books_lines.append(f"{i + 1}. \"{title}\" — {desc}{'...' if desc else ''}")

    books_context: str = "\n".join(books_lines)
    prompt: str = _EVALUATE_PROMPT.format(query=query, books_context=books_context)

    # ── 5. LLM call: timeout + retry on transient errors ─────────────────────
    t_start = time.monotonic()
    eval_result: Optional[EvaluationResult] = None

    for attempt in range(2):          # attempt 0 (first try) + attempt 1 (retry)
        try:
            llm = LLMFactory.create(
                provider=provider,
                api_key=api_key,
                temperature=0.1,
            )
            eval_result = await asyncio.wait_for(
                _invoke_with_fallback(llm, prompt),
                timeout=timeout_s,
            )
            break                      # success, exit retry loop

        except asyncio.TimeoutError:
            logger.warning("LLM Evaluate [%s]: timeout after %.1fs → rule fallback",
                           evaluation_id, timeout_s)
            _hist, _deg, _cache = _get_prom_metrics()
            if _deg is not None:
                _deg.labels(reason="timeout").inc()
            return _rule_based_evaluate(state, reason="llm_timeout")

        except Exception as exc:       # noqa: BLE001
            # Check if this is a retryable error (429 / 5xx)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 0 and status in _RETRYABLE_STATUS:
                delay = 0.5 * (2 ** attempt)
                logger.warning(
                    "LLM Evaluate [%s]: HTTP %s, retrying in %.1fs (attempt %d/2)",
                    evaluation_id, status, delay, attempt + 1,
                )
                await asyncio.sleep(delay)
                continue

            logger.error("LLM Evaluate [%s]: %s → rule fallback", evaluation_id, exc)
            _hist, _deg, _cache = _get_prom_metrics()
            if _deg is not None:
                _deg.labels(reason=f"error:{type(exc).__name__}").inc()
            return _rule_based_evaluate(state, reason=f"llm_error:{type(exc).__name__}")

    if eval_result is None:
        return _rule_based_evaluate(state, reason="llm_error:no_result")

    latency_ms = (time.monotonic() - t_start) * 1000
    quality_score = eval_result.score
    quality_reason = eval_result.reason

    # ── 6. Structured evaluation log (A/B experiment tracing) ────────────────
    logger.info(
        "LLM Evaluate | id=%s strategy=%s score=%.2f threshold=%.2f "
        "need_more=%s latency_ms=%.0f provider=%s failed=False",
        evaluation_id,
        state.get("strategy", "unknown"),
        quality_score,
        threshold,
        quality_score < threshold,
        latency_ms,
        provider,
    )

    # ── 7. Build result, record metrics, populate cache ───────────────────────
    result = {
        "need_more": quality_score < threshold,
        "quality_score": quality_score,
        "quality_reason": quality_reason,
        "llm_evaluate_failed": False,
    }
    _hist, _deg, _cache = _get_prom_metrics()
    if _hist is not None:
        _hist.labels(provider=provider, result="success").observe(latency_ms / 1000)
    llm_evaluate_cache.set(query, isbn_list, max_books, result)
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _invoke_with_fallback(llm, prompt: str) -> EvaluationResult:
    """
    Two-path LLM invocation:
      Path A: with_structured_output (OpenAI, Groq, modern Ollama with tool support)
      Path B: raw ainvoke + JSON regex parse (older Ollama, any provider as fallback)
    """
    # Path A: structured output (OpenAI / Groq / modern providers)
    try:
        structured = llm.with_structured_output(EvaluationResult)
        result = await structured.ainvoke(prompt)
        if isinstance(result, EvaluationResult):
            return result
        return EvaluationResult(**result)
    except (NotImplementedError, AttributeError, ValidationError, TypeError) as exc:
        # Provider does not support response_format → expected, fall to Path B
        logger.debug("_invoke_with_fallback: Path A declined (%s: %s), trying Path B",
                     type(exc).__name__, exc)
    except Exception as exc:  # noqa: BLE001
        # API-level rejection (e.g. DeepSeek 400 "response_format unavailable")
        # Log and fall to Path B; do NOT swallow logic errors silently
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status and status < 400:
            raise   # unexpected non-API error → re-raise for outer handler
        logger.debug("_invoke_with_fallback: Path A API error %s (%s), trying Path B",
                     status, exc)

    # Path B
    response = await llm.ainvoke(prompt)
    raw_text: str = getattr(response, "content", str(response))
    return _parse_json_response(raw_text)


def _parse_json_response(text: str) -> EvaluationResult:
    """
    Extract EvaluationResult from raw LLM text.
    Attempts strict JSON parse first, then regex extraction.
    Score is clamped to [0.0, 1.0] to guard against out-of-range LLM output.
    """
    def _build(data: dict) -> EvaluationResult:
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))          # clamp
        reason = str(data.get("reason", ""))[:300]
        return EvaluationResult(score=score, reason=reason)

    # Attempt 1: direct JSON parse
    try:
        return _build(json.loads(text.strip()))
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Attempt 2: extract first {...} block containing "score"
    match = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return _build(json.loads(match.group()))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    raise ValueError(f"Cannot parse LLM output as EvaluationResult: {text[:300]!r}")


def _rule_based_evaluate(state: RAGState, reason: str) -> Dict[str, Any]:
    """
    Fallback to rule-based evaluation (identical logic to evaluate_node).
    Called when LLM times out, errors, or isbn_list is insufficient.
    """
    n_results = len(state.get("isbn_list", []))
    freshness_fallback = state.get("freshness_fallback", False)
    threshold = state.get("freshness_threshold", 3)

    if n_results == 0 and freshness_fallback:
        need_more = True
    elif n_results < threshold and freshness_fallback:
        need_more = True
    elif n_results < 2:
        need_more = True
    else:
        need_more = False

    return {
        "need_more": need_more,
        "quality_score": None,
        "quality_reason": f"rule_fallback:{reason}",
        "llm_evaluate_failed": True,
    }


# ── Existing async web fallback (unchanged) ───────────────────────────────────

async def web_fallback_node(state: RAGState, config=None) -> Dict[str, Any]:
    """
    Fetch from Google Books API when local results insufficient (async).
    Uses search_google_books_async to avoid blocking the event loop.
    """
    from src.rag.web_search import search_google_books_async
    from src.data.stores.metadata_store import metadata_store

    query = state["query"]
    category = state.get("category", "All")
    existing_isbns = set(state.get("isbn_list", []))
    max_to_fetch = 10 - len(existing_isbns)

    if max_to_fetch <= 0:
        return {"need_more": False}

    recommender = None
    if config:
        cfg = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {}) or {}
        recommender = cfg.get("recommender") if cfg else None

    web_books = await search_google_books_async(query, max_results=max_to_fetch * 2)
    new_isbns = list(existing_isbns)

    for book in web_books:
        isbn = book.get("isbn13", "")
        if not isbn or isbn in existing_isbns:
            continue
        if metadata_store.book_exists(isbn):
            continue
        if category and category != "All":
            book_cat = book.get("simple_categories", "")
            if category.lower() not in (book_cat or "").lower():
                continue

        if recommender:
            added = recommender.add_new_book(
                isbn=isbn,
                title=book.get("title", ""),
                author=book.get("authors", "Unknown"),
                description=book.get("description", ""),
                category=book.get("simple_categories", "General"),
                thumbnail=book.get("thumbnail"),
                published_date=book.get("publishedDate", ""),
            )
            if added:
                new_isbns.append(isbn)
        else:
            new_isbns.append(isbn)

        if len(new_isbns) - len(existing_isbns) >= max_to_fetch:
            break

    logger.info(f"Agentic Web Fallback: added {len(new_isbns) - len(existing_isbns)} books")
    return {"isbn_list": new_isbns, "need_more": False, "retry_count": 1}
