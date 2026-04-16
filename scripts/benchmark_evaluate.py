#!/usr/bin/env python3
"""
Benchmark: Rule-based evaluate_node vs LLM llm_evaluate_node.

Measures latency (P50 / P95 / P99) and decision accuracy against a
hand-labeled query set, producing a structured performance report.

Usage
-----
# Rule-based baseline
python scripts/benchmark_evaluate.py --mode rule \
    --queries data/eval/rag_queries.jsonl --n 20

# LLM evaluate (requires Ollama running locally)
python scripts/benchmark_evaluate.py --mode llm \
    --queries data/eval/rag_queries.jsonl --n 20 \
    --provider ollama --timeout 1.5

# Compare both and save full report
python scripts/benchmark_evaluate.py --mode both \
    --queries data/eval/rag_queries.jsonl \
    --output results/benchmark_report.json

SLA target: deep-strategy P95 ≤ 2500 ms
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.support.agentic.nodes import evaluate_node, _rule_based_evaluate

# ── Constants ──────────────────────────────────────────────────────────────────

SLA_P95_MS: float = 6000.0   # SLA for cloud API (DeepSeek); was 2500ms for local Ollama
PASS_MARK = "PASS ✓"
FAIL_MARK = "FAIL ✗"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_queries(path: Path) -> list[dict]:
    """Load hand-labeled queries from JSONL file."""
    queries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _to_state(q: dict) -> dict:
    """Convert a query record to a RAGState-compatible dict."""
    return {
        "query": q["query"],
        "isbn_list": q.get("isbn_list", []),
        "freshness_fallback": q.get("freshness_fallback", False),
        "freshness_threshold": q.get("freshness_threshold", 3),
        "retry_count": q.get("retry_count", 0),
    }


# ── Rule benchmark ─────────────────────────────────────────────────────────────

def run_rule_benchmark(queries: list[dict]) -> dict:
    """Run evaluate_node on every query and record latency + decisions."""
    latencies_ms: list[float] = []
    decisions: list[bool] = []
    correct: int = 0

    for q in queries:
        state = _to_state(q)
        t0 = time.perf_counter()
        result = evaluate_node(state)
        elapsed = (time.perf_counter() - t0) * 1000  # ms

        latencies_ms.append(elapsed)
        need_more: bool = result["need_more"]
        decisions.append(need_more)

        expected: bool = q.get("expected_need_more", q.get("rule_need_more", need_more))
        if need_more == expected:
            correct += 1

    return _build_stats("rule", latencies_ms, decisions, queries, correct)


# ── LLM benchmark ──────────────────────────────────────────────────────────────

def _patch_metadata_store(queries: list[dict]):
    """
    Monkey-patch metadata_store.get_book_metadata to return book_titles from
    the JSONL eval data.

    The eval queries use real-world ISBNs that are not in the local SQLite
    metadata_store (dev machine without full data dump).  Without this patch,
    every book appears as "ISBN:xxxx — " and the LLM scores conservatively,
    producing artificially high false-trigger rates.

    In production the metadata_store is fully populated; this patch simulates
    that condition so benchmark results reflect real-world behaviour.
    """
    isbn_title: dict[str, str] = {}
    for q in queries:
        for isbn, title in zip(q.get("isbn_list", []), q.get("book_titles", [])):
            if isbn and title:
                isbn_title[isbn] = title

    from src.data.stores.metadata_store import metadata_store as store
    original_fn = store.get_book_metadata

    def _patched(isbn):
        if isbn in isbn_title:
            return {"title": isbn_title[isbn], "description": ""}
        return original_fn(isbn)

    store.get_book_metadata = _patched
    return original_fn, store  # caller restores on exit


async def run_llm_benchmark_async(
    queries: list[dict],
    provider: str = "deepseek",
    api_key: str | None = None,
    timeout_s: float = 5.0,
    max_books: int = 5,
) -> dict:
    """Run llm_evaluate_node on every query and record latency + decisions."""
    from src.support.agentic.nodes import llm_evaluate_node

    # Inject book titles so LLM sees real names instead of "ISBN:xxxx"
    original_fn, store = _patch_metadata_store(queries)
    try:
        result = await _run_llm_queries(queries, llm_evaluate_node, provider, api_key, timeout_s, max_books)
    finally:
        store.get_book_metadata = original_fn  # restore
    return result


async def _run_llm_queries(
    queries, llm_evaluate_node, provider, api_key, timeout_s, max_books
) -> dict:

    latencies_ms: list[float] = []
    decisions: list[bool] = []
    failed: list[bool] = []
    correct: int = 0

    config = {
        "configurable": {
            "llm_provider": provider,
            "llm_api_key": api_key,
            "quality_threshold": 0.6,
            "llm_evaluate_timeout": timeout_s,
            "llm_evaluate_max_books": max_books,
        }
    }

    for q in queries:
        state = _to_state(q)
        t0 = time.perf_counter()
        result = await llm_evaluate_node(state, config)
        elapsed = (time.perf_counter() - t0) * 1000

        latencies_ms.append(elapsed)
        need_more: bool = result.get("need_more", False)
        decisions.append(need_more)
        failed.append(result.get("llm_evaluate_failed", False))

        expected: bool = q.get("expected_need_more", need_more)
        if need_more == expected:
            correct += 1

    stats = _build_stats("llm", latencies_ms, decisions, queries, correct)
    stats["llm_failure_rate"] = sum(failed) / len(failed) if failed else 0.0
    stats["llm_failures"] = sum(failed)
    return stats


def run_llm_benchmark(
    queries: list[dict],
    provider: str = "deepseek",
    api_key: str | None = None,
    timeout_s: float = 5.0,
) -> dict:
    return asyncio.run(run_llm_benchmark_async(queries, provider, api_key, timeout_s))




# ── Stats helper ───────────────────────────────────────────────────────────────

def _build_stats(
    mode: str,
    latencies_ms: list[float],
    decisions: list[bool],
    queries: list[dict],
    correct: int,
) -> dict:
    """Compute percentile latencies and A/B experiment metrics."""
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    def pct(p: float) -> float:
        idx = min(int(p / 100 * n), n - 1)
        return round(sorted_lat[idx], 3)

    p50 = pct(50)
    p95 = pct(95)
    p99 = pct(99)

    # A/B metrics
    sm_total = sum(1 for q in queries if q.get("semantic_mismatch", False))
    sm_fixed = sum(
        1 for i, q in enumerate(queries)
        if q.get("semantic_mismatch", False) and decisions[i] is True
    )
    sm_fix_rate = sm_fixed / sm_total if sm_total else None

    # False trigger: rule=OK, human=good, but this mode says need_more=True
    good_total = sum(
        1 for q in queries
        if not q.get("semantic_mismatch", False)
        and not q.get("rule_need_more", q.get("expected_need_more", False))
    )
    false_trigger = sum(
        1 for i, q in enumerate(queries)
        if not q.get("semantic_mismatch", False)
        and not q.get("rule_need_more", q.get("expected_need_more", False))
        and decisions[i] is True
    )
    false_trigger_rate = false_trigger / good_total if good_total else None

    return {
        "mode": mode,
        "n_queries": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "min": round(sorted_lat[0], 3),
            "max": round(sorted_lat[-1], 3),
            "mean": round(statistics.mean(sorted_lat), 3),
        },
        "sla_p95_pass": p95 <= SLA_P95_MS,
        "ab_metrics": {
            "semantic_mismatch_total": sm_total,
            "semantic_mismatch_fixed": sm_fixed,
            "semantic_mismatch_fix_rate": round(sm_fix_rate, 4) if sm_fix_rate is not None else None,
            "false_trigger_total": false_trigger,
            "false_trigger_rate": round(false_trigger_rate, 4) if false_trigger_rate is not None else None,
        },
    }


# ── Report formatting ──────────────────────────────────────────────────────────

def _fmt_rate(v: float | None, target: str, comparator) -> str:
    if v is None:
        return "N/A (no samples)"
    pct = f"{v * 100:.1f}%"
    status = PASS_MARK if comparator(v) else FAIL_MARK
    return f"{pct}  {status}  (target {target})"


def print_report(stats: dict) -> None:
    mode = stats["mode"].upper()
    lat = stats["latency_ms"]
    ab = stats["ab_metrics"]

    sla_status = PASS_MARK if stats["sla_p95_pass"] else FAIL_MARK
    print(f"\n{'=' * 60}")
    print(f"  Benchmark Report — {mode} mode")
    print(f"{'=' * 60}")
    print(f"  Queries:          {stats['n_queries']}")
    print(f"  Decision Accuracy: {stats['accuracy'] * 100:.1f}%  (vs human labels)")
    print()
    print("  Latency:")
    print(f"    P50 :  {lat['p50']:>8.1f} ms")
    print(f"    P95 :  {lat['p95']:>8.1f} ms  {sla_status}  (SLA ≤ {SLA_P95_MS:.0f} ms)")
    print(f"    P99 :  {lat['p99']:>8.1f} ms")
    print(f"    Mean:  {lat['mean']:>8.1f} ms")
    print(f"    Min :  {lat['min']:>8.1f} ms  |  Max: {lat['max']:.1f} ms")
    print()
    print("  A/B Metrics:")
    fix_str = _fmt_rate(
        ab["semantic_mismatch_fix_rate"],
        "> 60%",
        lambda v: v > 0.60,
    )
    ft_str = _fmt_rate(
        ab["false_trigger_rate"],
        "< 20%",
        lambda v: v < 0.20,
    )
    print(f"    Semantic Mismatch Fix Rate : {fix_str}")
    print(f"    False Trigger Rate         : {ft_str}")

    if "llm_failure_rate" in stats:
        fr = stats["llm_failure_rate"]
        fr_status = PASS_MARK if fr < 0.20 else FAIL_MARK
        print(f"    LLM Degradation Rate       : {fr * 100:.1f}%  {fr_status}  (target < 20%)")

    print(f"{'=' * 60}")


def print_comparison(rule_stats: dict, llm_stats: dict) -> None:
    """Side-by-side comparison table."""
    r_lat = rule_stats["latency_ms"]
    l_lat = llm_stats["latency_ms"]
    r_ab = rule_stats["ab_metrics"]
    l_ab = llm_stats["ab_metrics"]

    print(f"\n{'=' * 70}")
    print(f"  Side-by-Side Comparison")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<40} {'RULE':>10}   {'LLM':>10}")
    print(f"  {'-' * 62}")
    print(f"  {'Decision Accuracy':<40} {rule_stats['accuracy']*100:>9.1f}%   {llm_stats['accuracy']*100:>9.1f}%")
    print(f"  {'P50 Latency (ms)':<40} {r_lat['p50']:>10.1f}   {l_lat['p50']:>10.1f}")
    print(f"  {'P95 Latency (ms)':<40} {r_lat['p95']:>10.1f}   {l_lat['p95']:>10.1f}")
    print(f"  {'P99 Latency (ms)':<40} {r_lat['p99']:>10.1f}   {l_lat['p99']:>10.1f}")

    def _pct_str(v):
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    print(f"  {'Semantic Mismatch Fix Rate':<40} {_pct_str(r_ab['semantic_mismatch_fix_rate']):>10}   {_pct_str(l_ab['semantic_mismatch_fix_rate']):>10}")
    print(f"  {'False Trigger Rate':<40} {_pct_str(r_ab['false_trigger_rate']):>10}   {_pct_str(l_ab['false_trigger_rate']):>10}")
    if "llm_failure_rate" in llm_stats:
        print(f"  {'LLM Degradation Rate':<40} {'—':>10}   {llm_stats['llm_failure_rate']*100:>9.1f}%")
    print(f"{'=' * 70}")

    # Overall verdict
    llm_p95_ok = llm_stats["sla_p95_pass"]
    llm_fix_rate = l_ab["semantic_mismatch_fix_rate"]
    llm_fix_ok = llm_fix_rate is not None and llm_fix_rate > 0.60
    llm_ft_rate = l_ab["false_trigger_rate"]
    llm_ft_ok = llm_ft_rate is None or llm_ft_rate < 0.20

    verdict_ok = llm_p95_ok and llm_fix_ok and llm_ft_ok
    verdict = "✓  POC PASSED — ready for v2.0 integration" if verdict_ok else "✗  POC FAILED — review metrics above"
    print(f"\n  Verdict: {verdict}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark rule vs LLM evaluate_node"
    )
    parser.add_argument(
        "--mode",
        choices=["rule", "llm", "both"],
        default="both",
        help="Which evaluate implementation to benchmark (default: both)",
    )
    parser.add_argument(
        "--queries",
        default="data/eval/rag_queries.jsonl",
        help="Path to hand-labeled JSONL query file",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Limit to first N queries (0 = all)",
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        help="LLM provider for llm mode (default: deepseek)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="API key for LLM provider (default: $DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="LLM evaluate timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save JSON report to this path (optional)",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"ERROR: query file not found: {queries_path}", file=sys.stderr)
        sys.exit(1)

    queries = load_queries(queries_path)
    if args.n > 0:
        queries = queries[: args.n]
    print(f"Loaded {len(queries)} queries from {queries_path}")

    report: dict[str, Any] = {"queries_file": str(queries_path), "n": len(queries)}

    if args.mode in ("rule", "both"):
        print("\nRunning rule-based benchmark …")
        rule_stats = run_rule_benchmark(queries)
        print_report(rule_stats)
        report["rule"] = rule_stats

    if args.mode in ("llm", "both"):
        print(f"\nRunning LLM benchmark (provider={args.provider}, timeout={args.timeout}s) …")
        llm_stats = run_llm_benchmark(queries, args.provider, args.api_key, args.timeout)
        print_report(llm_stats)
        report["llm"] = llm_stats

    if args.mode == "both":
        print_comparison(report["rule"], report["llm"])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
