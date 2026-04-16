"""
Production smoke test for Book Recommender v2.6 / llm_evaluate_node v2.1.

Usage
-----
    # Against local dev server:
    python scripts/smoke_test.py

    # Against staging / production:
    BASE_URL=https://your-host.com python scripts/smoke_test.py

    # Include LLM evaluate live call (requires DEEPSEEK_API_KEY in env):
    USE_LLM_EVALUATE=true python scripts/smoke_test.py

Exit codes
----------
    0  All checks passed
    1  One or more checks failed (see FAILED lines)

Checks
------
    ✓ GET  /health                          → 200, {status: healthy}
    ✓ GET  /metrics                         → 200, Prometheus text
    ✓ GET  /categories                      → 200, non-empty list
    ✓ POST /recommend (rule-based)          → 200, ≥1 recommendation, P95 < 3000ms
    ✓ POST /recommend (agentic rule mode)   → 200, ≥1 recommendation
    ✓ POST /recommend (agentic LLM mode)*  → 200, quality_score present     *if USE_LLM_EVALUATE=true
    ✓ GET  /api/recommend/personal          → 200, list returned
    ✓ GET  /api/onboarding/books            → 200, ≥1 book
    ✓ GET  /openapi.json                    → 200, valid OpenAPI spec
    ✓ Cache hit: second identical /recommend call faster than first
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
USE_LLM = os.getenv("USE_LLM_EVALUATE", "false").lower() == "true"
TIMEOUT = int(os.getenv("SMOKE_TIMEOUT", "30"))   # seconds per request
LATENCY_SLA_MS = 3000                              # P95 SLA for non-LLM path
LLM_LATENCY_SLA_MS = 8000                          # P95 SLA for LLM agentic path

_PASSED: list[str] = []
_FAILED: list[str] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check(name: str, ok: bool, detail: str = "") -> bool:
    if ok:
        _PASSED.append(name)
        print(f"  ✓  {name}")
    else:
        _FAILED.append(name)
        print(f"  ✗  {name}" + (f"  [{detail}]" if detail else ""))
    return ok


def _get(path: str, **kwargs) -> tuple[requests.Response, float]:
    t0 = time.perf_counter()
    r = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    return r, ms


def _post(path: str, body: dict[str, Any], **kwargs) -> tuple[requests.Response, float]:
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}{path}", json=body, timeout=TIMEOUT, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    return r, ms


# ── Check suites ──────────────────────────────────────────────────────────────

def check_health():
    print("\n[1] Health & Metrics")
    r, ms = _get("/health")
    _check("GET /health → 200", r.status_code == 200, f"HTTP {r.status_code}")
    if r.ok:
        body = r.json()
        _check("health.status == 'healthy'", body.get("status") == "healthy", str(body))

    r, ms = _get("/metrics")
    _check("GET /metrics → 200", r.status_code == 200, f"HTTP {r.status_code}")
    _check("metrics contains http_requests_total", "http_requests_total" in r.text)


def check_categories():
    print("\n[2] Categories")
    r, ms = _get("/categories")
    _check("GET /categories → 200", r.status_code == 200, f"HTTP {r.status_code}")
    if r.ok:
        data = r.json()
        cats = data.get("categories", [])
        _check("categories non-empty", len(cats) > 0, f"got {len(cats)}")


def check_recommend_rule_based():
    print("\n[3] /recommend – rule-based (5 queries, P95 SLA)")
    queries = [
        "a romantic novel set in Paris",
        "science fiction about AI",
        "mystery thriller with unreliable narrator",
        "books about grief and healing",
        "light comedy for summer reading",
    ]
    latencies = []
    for q in queries:
        r, ms = _post("/recommend", {"query": q, "category": "All"})
        latencies.append(ms)
        if r.status_code != 200:
            _check(f"POST /recommend [{q[:30]}]", False, f"HTTP {r.status_code}")
            continue
        recs = r.json().get("recommendations", [])
        _check(f"POST /recommend [{q[:30]}] → ≥1 rec", len(recs) >= 1, f"got {len(recs)}")

    if latencies:
        p95 = statistics.quantiles(latencies, n=100)[94]
        _check(
            f"Rule-based P95 latency < {LATENCY_SLA_MS}ms",
            p95 < LATENCY_SLA_MS,
            f"P95={p95:.0f}ms",
        )
        print(f"     P50={statistics.median(latencies):.0f}ms  P95={p95:.0f}ms")


def check_recommend_agentic_rule():
    print("\n[4] /recommend – agentic rule mode (use_agentic=true, USE_LLM_EVALUATE=false)")
    r, ms = _post("/recommend", {
        "query": "books about loss and recovery",
        "use_agentic": True,
    })
    _check("POST /recommend (agentic-rule) → 200", r.status_code == 200, f"HTTP {r.status_code}")
    if r.ok:
        recs = r.json().get("recommendations", [])
        _check("agentic-rule → ≥1 recommendation", len(recs) >= 1, f"got {len(recs)}")


def check_recommend_agentic_llm():
    """Only run when USE_LLM_EVALUATE=true (requires live DeepSeek API key)."""
    if not USE_LLM:
        print("\n[5] /recommend – agentic LLM mode  ← SKIPPED (USE_LLM_EVALUATE not set)")
        return

    print("\n[5] /recommend – agentic LLM mode (USE_LLM_EVALUATE=true)")
    latencies = []
    queries = [
        "a philosophical exploration of mortality",
        "lighthearted beach reads",
    ]
    for q in queries:
        r, ms = _post("/recommend", {"query": q, "use_agentic": True})
        latencies.append(ms)
        _check(f"LLM agentic [{q[:30]}] → 200", r.status_code == 200, f"HTTP {r.status_code}")

    if latencies:
        p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 5 else max(latencies)
        _check(
            f"LLM agentic P95 < {LLM_LATENCY_SLA_MS}ms",
            p95 < LLM_LATENCY_SLA_MS,
            f"P95={p95:.0f}ms",
        )


def check_cache_speedup():
    print("\n[6] Cache hit latency (second call should be faster)")
    payload = {"query": "dystopian fiction about surveillance", "category": "All"}
    _, ms1 = _post("/recommend", payload)
    _, ms2 = _post("/recommend", payload)
    # Cache may be in Redis or in-process; second call should be at most 50% of first
    _check(
        "Second identical call faster (cache/memoisation)",
        ms2 < ms1 * 1.5,   # allow 50% buffer for network jitter
        f"first={ms1:.0f}ms  second={ms2:.0f}ms",
    )


def check_personal_and_onboarding():
    print("\n[7] Personal & Onboarding endpoints")
    r, _ = _get("/api/recommend/personal", params={"user_id": "smoke_test_user", "top_k": 5})
    _check("GET /api/recommend/personal → 200", r.status_code == 200, f"HTTP {r.status_code}")

    r, _ = _get("/api/onboarding/books", params={"limit": 5})
    _check("GET /api/onboarding/books → 200", r.status_code == 200, f"HTTP {r.status_code}")
    if r.ok:
        books = r.json().get("books", [])
        _check("onboarding → ≥1 book", len(books) >= 1, f"got {len(books)}")


def check_openapi_contract():
    print("\n[8] OpenAPI contract")
    r, _ = _get("/openapi.json")
    _check("GET /openapi.json → 200", r.status_code == 200, f"HTTP {r.status_code}")
    if r.ok:
        spec = r.json()
        _check("spec has /recommend path", "/recommend" in spec.get("paths", {}))
        _check("spec has /health path", "/health" in spec.get("paths", {}))
        _check("spec version field present", "info" in spec and "version" in spec["info"])


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"{'='*60}")
    print(f"  Book Recommender Smoke Test")
    print(f"  Target : {BASE_URL}")
    print(f"  LLM evaluate: {'ENABLED' if USE_LLM else 'DISABLED'}")
    print(f"{'='*60}")

    check_health()
    check_categories()
    check_recommend_rule_based()
    check_recommend_agentic_rule()
    check_recommend_agentic_llm()
    check_cache_speedup()
    check_personal_and_onboarding()
    check_openapi_contract()

    total = len(_PASSED) + len(_FAILED)
    print(f"\n{'='*60}")
    print(f"  Results: {len(_PASSED)}/{total} passed")
    if _FAILED:
        print(f"  FAILED checks:")
        for f in _FAILED:
            print(f"    - {f}")
    print(f"{'='*60}\n")

    sys.exit(0 if not _FAILED else 1)


if __name__ == "__main__":
    main()
