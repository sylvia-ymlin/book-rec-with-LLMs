"""
Quick test for concurrent benchmark logic without loading full recommender.
Run: python benchmarks/test_concurrent_benchmark.py
"""

import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock recommender that simulates ~100ms latency
class MockRecommender:
    def get_recommendations_sync(self, query: str, category: str = "All", tone: str = "All"):
        time.sleep(0.1)
        return [{"title": "Mock Book", "isbn": "123"}]

TEST_QUERIES = ["query A", "query B", "query C"]


def _run_one_query(recommender, query: str) -> tuple[float, int]:
    start = time.perf_counter()
    recommender.get_recommendations_sync(query, category="All", tone="All")
    return (time.perf_counter() - start) * 1000, 1


def benchmark_concurrent(recommender, n_workers: int = 5, total_queries: int = 15) -> dict:
    queries = [TEST_QUERIES[i % len(TEST_QUERIES)] for i in range(total_queries)]
    latencies = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_run_one_query, recommender, q) for q in queries]
        for future in as_completed(futures):
            lat_ms, _ = future.result()
            latencies.append(lat_ms)

    wall_sec = time.perf_counter() - start

    return {
        "operation": f"Concurrent ({n_workers} workers)",
        "workers": n_workers,
        "total_queries": total_queries,
        "wall_sec": round(wall_sec, 2),
        "qps": round(total_queries / wall_sec, 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
    }


def main():
    mock = MockRecommender()

    # Sequential: 15 * 100ms = ~1.5s
    print("Sequential (1 worker):")
    r1 = benchmark_concurrent(mock, n_workers=1, total_queries=15)
    print(f"  wall_sec={r1['wall_sec']}, qps={r1['qps']}, mean_ms={r1['mean_latency_ms']}")

    # Concurrent: 15 queries with 5 workers -> ~3 batches of 5 -> ~300ms
    print("\nConcurrent (5 workers):")
    r5 = benchmark_concurrent(mock, n_workers=5, total_queries=15)
    print(f"  wall_sec={r5['wall_sec']}, qps={r5['qps']}, mean_ms={r5['mean_latency_ms']}")

    # Concurrency should give ~5x speedup
    speedup = r1["wall_sec"] / r5["wall_sec"]
    print(f"\nSpeedup: {speedup:.1f}x (expected ~5x for 5 workers)")
    assert r5["qps"] > r1["qps"], "Concurrent QPS should exceed sequential"
    print("OK: Concurrent benchmark logic works correctly.")


if __name__ == "__main__":
    main()
