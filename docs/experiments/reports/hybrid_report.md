# Hybrid Retrieval Benchmark Report
**Date**: 2026-01-08
**Metric**: Qualtitative Recall (Top-5)

## Experiment Setup
- **System**: Hybrid RRF (BM25 + Chroma Dense).
- **Comparison**: Baseline (Dense Only) vs Hybrid.

## Results Comparison

| Query Type | Query | Baseline Result | Hybrid Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Semantic** | "finding love..." | "All About Love" | "Elusive Love", "Finding God..." | ✅ **Maintained** |
| **Keyword** | "Harry Potter" | "Harry Potter and Philosophy" | **"Harry Potter and the Sorcerer's Stone"** | 🚀 **IMPROVED** |
| **Exact** | "0060959479" | "National Geographic..." (Fail) | **"All About Love: New Visions"** | 🎉 **FIXED** |

## Performance Trade-off
- **Latency**: Increased from ~20ms (Dense) to ~600ms (Hybrid).
- **Cause**: In-memory BM25 scoring of 220k documents in Python.
- **Verdict**: Acceptable for "High Accuracy" mode.

## Technical Implementation
- **Sparse**: `rank_bm25` (Okapi BM25) on Title + Author + Desc + ISBN.
- **Dense**: `all-MiniLM-L6-v2` (Chroma).
- **Fusion**: Reciprocal Rank Fusion (RRF) with `k=60`.

## Conclusion
Hybrid Search successfully combines the "Literal Precision" of BM25 with the "Semantic Understanding" of Vectors. We have solved the "Exact Match" failure case.
