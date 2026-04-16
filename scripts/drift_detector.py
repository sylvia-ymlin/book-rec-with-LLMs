"""
LLM Evaluate — 模型性能监控 & 数据漂移检测

功能
----
1. 分数分布分析：当前 score 分布 vs 基准分布（PSI 检测）
2. 系统性失败查询检测：哪些 query 类型始终触发 web_fallback
3. 降级率时序分析：从 Prometheus 拉取历史数据
4. 自动生成监控报告（Markdown）

使用
----
    # 对已有 benchmark 结果文件进行漂移分析
    python scripts/drift_detector.py --input data/eval/rag_queries.jsonl

    # 从 Prometheus 拉取实时数据（需要 Prometheus 运行）
    python scripts/drift_detector.py --prometheus http://localhost:9090

    # 完整模式：benchmark + Prometheus + 报告输出
    python scripts/drift_detector.py \\
        --input data/eval/rag_queries.jsonl \\
        --prometheus http://localhost:9090 \\
        --output docs/ops/monitoring-report-$(date +%Y%m%d).md

PSI（Population Stability Index）判断标准
------------------------------------------
    PSI < 0.10  → 分布稳定（绿灯）
    0.10–0.25   → 轻微漂移（黄灯，观察）
    PSI > 0.25  → 显著漂移（红灯，需要重新评估 threshold）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 基准分布（来自 2026-03-23 benchmark 运行结果）─────────────────────────────
# Bucket 边界：[0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
BASELINE_SCORE_DIST = {
    "0.0-0.2": 0.05,
    "0.2-0.4": 0.15,
    "0.4-0.6": 0.20,
    "0.6-0.8": 0.30,
    "0.8-1.0": 0.30,
}

BASELINE_DEGRADATION_RATE = 0.20   # 20% from initial benchmark
BASELINE_SEMANTIC_FIX_RATE = 0.857  # 85.7%
BASELINE_FALSE_TRIGGER_RATE = 0.222  # 22.2%

PSI_WARN = 0.10
PSI_ALERT = 0.25


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bucket(score: float) -> str:
    if score < 0.2:
        return "0.0-0.2"
    elif score < 0.4:
        return "0.2-0.4"
    elif score < 0.6:
        return "0.4-0.6"
    elif score < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def _psi(baseline: dict[str, float], current: dict[str, float]) -> float:
    """Population Stability Index between two probability distributions."""
    eps = 1e-6
    psi_val = 0.0
    for bucket in baseline:
        b = baseline.get(bucket, eps)
        c = current.get(bucket, eps)
        psi_val += (c - b) * math.log((c + eps) / (b + eps))
    return psi_val


def _psi_status(psi_val: float) -> str:
    if psi_val < PSI_WARN:
        return "🟢 STABLE"
    elif psi_val < PSI_ALERT:
        return "🟡 WARN"
    return "🔴 DRIFT"


def _fetch_prometheus(base_url: str, query: str, step: str = "5m") -> list[dict]:
    """Fetch instant query result from Prometheus HTTP API."""
    try:
        import urllib.request
        import urllib.parse
        params = urllib.parse.urlencode({"query": query})
        url = f"{base_url}/api/v1/query?{params}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())["data"]["result"]
    except Exception as exc:
        print(f"  ⚠ Prometheus query failed: {exc}", file=sys.stderr)
        return []


# ── Analysis functions ────────────────────────────────────────────────────────

def analyze_score_distribution(queries: list[dict]) -> dict[str, Any]:
    """Compute current score distribution and PSI vs baseline."""
    scored = [q for q in queries if q.get("llm_score") is not None]
    if not scored:
        return {"status": "NO_LLM_SCORES", "psi": None}

    bucket_counts: dict[str, int] = defaultdict(int)
    for q in scored:
        bucket_counts[_bucket(float(q["llm_score"]))] += 1

    total = len(scored)
    current_dist = {k: v / total for k, v in bucket_counts.items()}

    # Fill missing buckets with epsilon
    for b in BASELINE_SCORE_DIST:
        current_dist.setdefault(b, 1e-6)

    psi_val = _psi(BASELINE_SCORE_DIST, current_dist)
    return {
        "n_scored": total,
        "distribution": current_dist,
        "baseline": BASELINE_SCORE_DIST,
        "psi": round(psi_val, 4),
        "status": _psi_status(psi_val),
    }


def analyze_systematic_failures(queries: list[dict]) -> list[dict]:
    """Find query patterns that consistently trigger web_fallback."""
    # Group by strategy
    strategy_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "need_more": 0})
    for q in queries:
        strat = q.get("strategy", "unknown")
        strategy_stats[strat]["total"] += 1
        if q.get("expected_need_more", False):
            strategy_stats[strat]["need_more"] += 1

    failures = []
    for strat, stats in strategy_stats.items():
        rate = stats["need_more"] / max(stats["total"], 1)
        if rate > 0.5:
            failures.append({
                "strategy": strat,
                "need_more_rate": round(rate, 2),
                "total": stats["total"],
                "recommendation": f"Strategy '{strat}' has {rate:.0%} need_more rate → "
                                  f"consider tuning vector search or lowering threshold",
            })

    # Find semantic mismatch patterns
    mismatch_queries = [q for q in queries if q.get("semantic_mismatch")]
    mismatch_keywords: dict[str, int] = defaultdict(int)
    for q in mismatch_queries:
        words = q.get("query", "").lower().split()
        for w in words:
            if len(w) > 4:  # skip short words
                mismatch_keywords[w] += 1

    top_mismatch_words = sorted(
        mismatch_keywords.items(), key=lambda x: -x[1]
    )[:5]

    return {
        "by_strategy": failures,
        "top_mismatch_keywords": top_mismatch_words,
        "total_mismatch": len(mismatch_queries),
        "total_queries": len(queries),
        "mismatch_rate": round(len(mismatch_queries) / max(len(queries), 1), 2),
    }


def analyze_prometheus_metrics(base_url: str) -> dict[str, Any]:
    """Fetch and analyze live Prometheus metrics."""
    print(f"  Querying Prometheus at {base_url}...")

    results = {}

    # Degradation rate (5m window)
    deg_rate = _fetch_prometheus(
        base_url,
        "rate(llm_evaluate_degradation_total[5m]) / "
        "(rate(llm_evaluate_latency_seconds_count[5m]) + rate(llm_evaluate_cache_hits_total[5m]) + 0.001)"
    )
    if deg_rate:
        val = float(deg_rate[0]["value"][1])
        results["live_degradation_rate"] = round(val, 3)
        results["degradation_status"] = (
            "🟢 OK" if val < 0.20 else "🟡 WARN" if val < 0.40 else "🔴 ALERT"
        )

    # P95 latency
    p95 = _fetch_prometheus(
        base_url,
        "histogram_quantile(0.95, sum(rate(llm_evaluate_latency_seconds_bucket[5m])) by (le))"
    )
    if p95:
        val = float(p95[0]["value"][1])
        results["live_p95_latency_s"] = round(val, 2)
        results["latency_status"] = (
            "🟢 OK" if val < 4 else "🟡 WARN" if val < 6 else "🔴 ALERT"
        )

    # Cache hit rate
    cache_hits = _fetch_prometheus(
        base_url,
        "rate(llm_evaluate_cache_hits_total[5m]) / "
        "(rate(llm_evaluate_latency_seconds_count[5m]) + rate(llm_evaluate_cache_hits_total[5m]) + 0.001)"
    )
    if cache_hits:
        val = float(cache_hits[0]["value"][1])
        results["live_cache_hit_rate"] = round(val, 3)

    # HTTP 5xx rate
    err_rate = _fetch_prometheus(
        base_url,
        'sum(rate(http_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))'
    )
    if err_rate:
        val = float(err_rate[0]["value"][1])
        results["live_5xx_rate"] = round(val, 4)
        results["error_status"] = (
            "🟢 OK" if val < 0.01 else "🟡 WARN" if val < 0.05 else "🔴 ALERT"
        )

    return results


def generate_report(
    score_analysis: dict,
    failure_analysis: dict,
    prom_analysis: dict,
    queries: list[dict],
    output_path: str | None,
) -> str:
    """Generate Markdown monitoring report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    semantic_mismatch_queries = [q for q in queries if q.get("semantic_mismatch")]
    llm_fix_count = sum(
        1 for q in semantic_mismatch_queries
        if q.get("llm_score") is not None and float(q.get("llm_score", 0)) < 0.6
    )
    fix_rate = llm_fix_count / max(len(semantic_mismatch_queries), 1)

    false_triggers = [
        q for q in queries
        if not q.get("expected_need_more") and q.get("llm_score") is not None
        and float(q.get("llm_score", 1)) < 0.6
    ]
    false_trigger_rate = len(false_triggers) / max(
        len([q for q in queries if not q.get("expected_need_more") and q.get("llm_score") is not None]), 1
    )

    lines = [
        f"# Book Recommender — 运维监控报告",
        f"",
        f"> 生成时间：{now} | 版本：v2.6-llm-evaluate-v2.1",
        f"",
        f"---",
        f"",
        f"## 1. 业务指标摘要",
        f"",
        f"| 指标 | 当前值 | 基准值 | 状态 |",
        f"|------|--------|--------|------|",
        f"| 语义不匹配修复率 | {fix_rate:.1%} | {BASELINE_SEMANTIC_FIX_RATE:.1%} | {'🟢 OK' if fix_rate >= 0.60 else '🔴 下降'} |",
        f"| 误触发率 | {false_trigger_rate:.1%} | {BASELINE_FALSE_TRIGGER_RATE:.1%} | {'🟢 OK' if false_trigger_rate <= 0.20 else '🟡 偏高'} |",
        f"| 评测样本数 | {len(queries)} | 20 | — |",
        f"",
        f"## 2. 分数分布 & 漂移检测（PSI）",
        f"",
    ]

    if score_analysis.get("psi") is not None:
        lines += [
            f"PSI = **{score_analysis['psi']}** → {score_analysis['status']}",
            f"",
            f"| 分桶 | 基准 | 当前 |",
            f"|------|------|------|",
        ]
        for b in BASELINE_SCORE_DIST:
            baseline_v = score_analysis["baseline"].get(b, 0)
            current_v = score_analysis["distribution"].get(b, 0)
            lines.append(f"| {b} | {baseline_v:.1%} | {current_v:.1%} |")
        lines.append("")
    else:
        lines.append("_无 LLM 分数数据（`llm_score` 字段缺失，需运行 benchmark_evaluate.py）_\n")

    lines += [
        f"## 3. 系统性失败查询分析",
        f"",
        f"- **语义不匹配率**：{failure_analysis['mismatch_rate']:.1%} ({failure_analysis['total_mismatch']}/{failure_analysis['total_queries']})",
        f"- **高频不匹配关键词**：{', '.join(w for w, _ in failure_analysis['top_mismatch_keywords'])}",
        f"",
    ]

    if failure_analysis["by_strategy"]:
        lines.append("**高 need_more 率策略**：\n")
        for f in failure_analysis["by_strategy"]:
            lines.append(f"- `{f['strategy']}` 策略：need_more = {f['need_more_rate']:.0%} — {f['recommendation']}")
        lines.append("")
    else:
        lines.append("_各策略 need_more 率均 < 50%，无系统性失败_\n")

    lines += [
        f"## 4. 实时 Prometheus 指标",
        f"",
    ]

    if prom_analysis:
        lines += [
            f"| 指标 | 值 | 状态 |",
            f"|------|----|------|",
        ]
        if "live_degradation_rate" in prom_analysis:
            lines.append(f"| LLM 降级率（5m） | {prom_analysis['live_degradation_rate']:.1%} | {prom_analysis.get('degradation_status', '—')} |")
        if "live_p95_latency_s" in prom_analysis:
            lines.append(f"| LLM P95 延迟 | {prom_analysis['live_p95_latency_s']:.2f}s | {prom_analysis.get('latency_status', '—')} |")
        if "live_cache_hit_rate" in prom_analysis:
            lines.append(f"| 缓存命中率（5m） | {prom_analysis['live_cache_hit_rate']:.1%} | — |")
        if "live_5xx_rate" in prom_analysis:
            lines.append(f"| HTTP 5xx 错误率 | {prom_analysis['live_5xx_rate']:.2%} | {prom_analysis.get('error_status', '—')} |")
        lines.append("")
    else:
        lines.append("_Prometheus 不可达或未配置 `--prometheus` 参数_\n")

    lines += [
        f"## 5. 待跟进问题",
        f"",
        f"| 优先级 | 问题 | 建议动作 |",
        f"|--------|------|---------|",
        f"| P0 | 多进程 worker 下 in-process 缓存失效 | 迁移到 Redis 缓存（v2.2 计划）|",
        f"| P1 | SQLite 未启用 WAL 模式 | 执行 `PRAGMA journal_mode=WAL` 初始化 |",
        f"| P2 | Prometheus metrics 使用私有 API | 已在 v2.1.1 修复 |",
        f"| P3 | 无用户点击反馈 | 添加 `/feedback` 端点（v2.2）|",
        f"",
        f"---",
        f"",
        f"*本报告由 `scripts/drift_detector.py` 自动生成*",
    ]

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")
        print(f"\n  报告已写入：{output_path}")

    return report


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluate 漂移检测 & 监控报告")
    parser.add_argument("--input", default="data/eval/rag_queries.jsonl",
                        help="评测数据集路径（JSONL）")
    parser.add_argument("--prometheus", default=None,
                        help="Prometheus base URL，例如 http://localhost:9090")
    parser.add_argument("--output", default=None,
                        help="报告输出路径（Markdown），默认打印到 stdout")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  LLM Evaluate 漂移检测")
    print(f"  数据集: {args.input}")
    print(f"  Prometheus: {args.prometheus or '(未配置)'}")
    print(f"{'='*60}\n")

    # 1. 加载数据集
    queries: list[dict] = []
    input_path = Path(args.input)
    if input_path.exists():
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
        print(f"  ✓ 加载 {len(queries)} 条查询记录")
    else:
        print(f"  ✗ 数据集文件不存在：{args.input}")
        queries = []

    # 2. 分数分布分析
    print("\n[1] 分数分布 & PSI 漂移检测...")
    score_analysis = analyze_score_distribution(queries)
    if score_analysis.get("psi") is not None:
        print(f"     PSI = {score_analysis['psi']} → {score_analysis['status']}")
    else:
        print(f"     {score_analysis['status']} — 无 llm_score 字段，跳过")

    # 3. 系统性失败分析
    print("\n[2] 系统性失败查询分析...")
    failure_analysis = analyze_systematic_failures(queries)
    print(f"     语义不匹配率: {failure_analysis['mismatch_rate']:.0%} "
          f"({failure_analysis['total_mismatch']}/{failure_analysis['total_queries']})")
    if failure_analysis["by_strategy"]:
        for f in failure_analysis["by_strategy"]:
            print(f"     ⚠ 策略 '{f['strategy']}': need_more = {f['need_more_rate']:.0%}")

    # 4. Prometheus 实时指标
    prom_analysis = {}
    if args.prometheus:
        print(f"\n[3] Prometheus 实时指标 ({args.prometheus})...")
        prom_analysis = analyze_prometheus_metrics(args.prometheus)
        for k, v in prom_analysis.items():
            if not k.endswith("_status"):
                print(f"     {k}: {v}")

    # 5. 生成报告
    print("\n[4] 生成监控报告...")
    report = generate_report(
        score_analysis, failure_analysis, prom_analysis,
        queries, args.output
    )

    if not args.output:
        print("\n" + "─" * 60)
        print(report)

    print(f"\n{'='*60}")
    print(f"  分析完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
