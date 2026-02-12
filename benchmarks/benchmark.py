"""
Performance Benchmark Script for Book Recommender System

This script measures:
1. Vector search latency
2. End-to-end recommendation latency
3. Throughput (queries per second, sequential)
4. Concurrent throughput (QPS under N parallel workers)

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --concurrent 5   # 5 concurrent workers

Note: For HTTP-level load testing (simulating real users), use Locust:
    pip install locust
    locust -f benchmarks/locustfile.py --host=http://localhost:8000
"""

import argparse
import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.recommendation_orchestrator import RecommendationOrchestrator
from src.vector_db import VectorDB


# Test queries for benchmarking
TEST_QUERIES = [
    "a romantic comedy set in New York",
    "a philosophical novel about the meaning of life",
    "a fast-paced thriller with plot twists",
    "a coming-of-age story about friendship and loss",
    "a historical fiction set during World War II",
    "a science fiction story about space exploration",
    "a mystery novel with an unreliable narrator",
    "a fantasy epic with dragons and magic",
    "a memoir about overcoming adversity",
    "a literary fiction exploring family dynamics",
]


def benchmark_vector_search(vector_db: VectorDB, n_runs: int = 50) -> dict:
    """Benchmark vector database search latency."""
    latencies = []
    
    for query in TEST_QUERIES:
        for _ in range(n_runs // len(TEST_QUERIES)):
            start = time.perf_counter()
            vector_db.search(query, k=50)
            latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        "operation": "Vector Search (k=50)",
        "runs": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "std_ms": round(statistics.stdev(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
    }


def benchmark_full_recommendation(recommender: RecommendationOrchestrator, n_runs: int = 30) -> dict:
    """Benchmark full recommendation pipeline latency."""
    latencies = []
    
    for query in TEST_QUERIES:
        for _ in range(n_runs // len(TEST_QUERIES)):
            start = time.perf_counter()
            recommender.get_recommendations_sync(query, category="All", tone="All")
            latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        "operation": "Full Recommendation",
        "runs": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "std_ms": round(statistics.stdev(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
    }


def benchmark_throughput(recommender: RecommendationOrchestrator, duration_sec: int = 10) -> dict:
    """Measure queries per second over a time window (sequential)."""
    query_count = 0
    start = time.perf_counter()
    query_idx = 0

    while (time.perf_counter() - start) < duration_sec:
        recommender.get_recommendations_sync(
            TEST_QUERIES[query_idx % len(TEST_QUERIES)],
            category="All",
            tone="All"
        )
        query_count += 1
        query_idx += 1

    elapsed = time.perf_counter() - start

    return {
        "operation": "Throughput Test (sequential)",
        "duration_sec": round(elapsed, 2),
        "total_queries": query_count,
        "qps": round(query_count / elapsed, 2),
    }


def _run_one_query(recommender: RecommendationOrchestrator, query: str) -> tuple[float, int]:
    """Run a single recommendation and return (latency_ms, 1)."""
    start = time.perf_counter()
    recommender.get_recommendations_sync(query, category="All", tone="All")
    return (time.perf_counter() - start) * 1000, 1


def benchmark_concurrent(
    recommender: RecommendationOrchestrator,
    n_workers: int = 5,
    total_queries: int = 50,
) -> dict:
    """
    Measure throughput under concurrent load using ThreadPoolExecutor.

    Simulates N parallel clients to expose:
    - VectorDB connection/query limits under load
    - GIL contention if CPU-bound (embedding, rerank)
    - I/O blocking in ChromaDB / LLM calls
    """
    queries = [TEST_QUERIES[i % len(TEST_QUERIES)] for i in range(total_queries)]
    latencies: list[float] = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_run_one_query, recommender, q) for q in queries
        ]
        for future in as_completed(futures):
            lat_ms, _ = future.result()
            latencies.append(lat_ms)

    wall_sec = time.perf_counter() - start

    return {
        "operation": f"Concurrent Throughput ({n_workers} workers)",
        "workers": n_workers,
        "total_queries": total_queries,
        "wall_sec": round(wall_sec, 2),
        "qps": round(total_queries / wall_sec, 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
    }


def print_results(results: list[dict]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    for result in results:
        print(f"\n📊 {result['operation']}")
        print("-" * 40)
        for key, value in result.items():
            if key != "operation":
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)


def save_results(results: list[dict], filepath: str = "benchmarks/results.md"):
    """Save results to markdown file."""
    with open(filepath, "w") as f:
        f.write("# Performance Benchmark Results\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## System Info\n")
        f.write("- Dataset: 5,000+ books\n")
        f.write("- Embedding Model: all-MiniLM-L6-v2 (384 dim)\n")
        f.write("- Vector DB: ChromaDB with HNSW index\n\n")
        
        f.write("## Results\n\n")
        
        for result in results:
            f.write(f"### {result['operation']}\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in result.items():
                if key != "operation":
                    f.write(f"| {key} | {value} |\n")
            f.write("\n")
        
        f.write("## Interpretation\n\n")
        f.write("- **Vector Search**: Time to query ChromaDB and retrieve top-k results\n")
        f.write("- **Full Recommendation**: End-to-end latency including filtering and formatting\n")
        f.write("- **Throughput (sequential)**: Sustained QPS when processing one query at a time\n")
        f.write("- **Concurrent Throughput**: QPS under N parallel workers; exposes GIL/IO bottlenecks\n")
    
    print(f"\n✅ Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Book Recommender System")
    parser.add_argument(
        "--concurrent",
        type=int,
        default=0,
        metavar="N",
        help="Add concurrent benchmark with N workers (e.g. 5). 0 = skip.",
    )
    parser.add_argument(
        "--concurrent-queries",
        type=int,
        default=50,
        help="Total queries for concurrent benchmark (default: 50)",
    )
    args = parser.parse_args()

    print("🚀 Initializing Book Recommender System...")
    print("   (This may take a moment to load models and vector database)")

    try:
        recommender = RecommendationOrchestrator()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    print("✅ System initialized. Starting benchmarks...\n")

    results = []

    # Benchmark 1: Vector Search
    print("📊 Running Vector Search benchmark...")
    results.append(benchmark_vector_search(recommender.vector_db))

    # Benchmark 2: Full Recommendation
    print("📊 Running Full Recommendation benchmark...")
    results.append(benchmark_full_recommendation(recommender))

    # Benchmark 3: Sequential Throughput
    print("📊 Running Sequential Throughput benchmark (10 seconds)...")
    results.append(benchmark_throughput(recommender))

    # Benchmark 4: Concurrent Throughput (optional)
    if args.concurrent > 0:
        print(f"📊 Running Concurrent Throughput ({args.concurrent} workers, {args.concurrent_queries} queries)...")
        results.append(
            benchmark_concurrent(
                recommender,
                n_workers=args.concurrent,
                total_queries=args.concurrent_queries,
            )
        )

    # Print and save results
    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
