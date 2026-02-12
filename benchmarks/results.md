# Performance Benchmark Results

**Date**: 2026-02-12 01:02:27

## System Info
- Dataset: 5,000+ books
- Embedding Model: all-MiniLM-L6-v2 (384 dim)
- Vector DB: ChromaDB with HNSW index

## Results

### Vector Search (k=50)

| Metric | Value |
|--------|-------|
| runs | 50 |
| mean_ms | 11.49 |
| median_ms | 6.43 |
| std_ms | 27.41 |
| min_ms | 5.49 |
| max_ms | 200.46 |
| p95_ms | 15.51 |

### Full Recommendation

| Metric | Value |
|--------|-------|
| runs | 30 |
| mean_ms | 3876.27 |
| median_ms | 260.87 |
| std_ms | 5445.93 |
| min_ms | 14.54 |
| max_ms | 16609.18 |
| p95_ms | 11694.41 |

### Throughput Test (sequential)

| Metric | Value |
|--------|-------|
| duration_sec | 10.1 |
| total_queries | 89 |
| qps | 8.81 |

### Concurrent Throughput (3 workers)

| Metric | Value |
|--------|-------|
| workers | 3 |
| total_queries | 12 |
| wall_sec | 1.29 |
| qps | 9.28 |
| mean_latency_ms | 298.3 |
| median_latency_ms | 370.19 |
| p95_latency_ms | 579.95 |

## Interpretation

- **Vector Search**: Time to query ChromaDB and retrieve top-k results
- **Full Recommendation**: End-to-end latency including filtering and formatting
- **Throughput (sequential)**: Sustained QPS when processing one query at a time
- **Concurrent Throughput**: QPS under N parallel workers; exposes GIL/IO bottlenecks
