# Reranking Benchmark Report
**Date**: 2026-01-08
**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Experiment Setup
- **Pipeline**: Hybrid Search (BM25 + Dense) -> Top 50 Candidates -> Cross-Encoder Rerank -> Top 5.
- **Metric**: Relevance Score & Qualitative Ranking.

## Results Comparison

| Query | Hybrid (Raw RRF) | Reranked Result (Top 1) | Score | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **"Harry Potter"** | "Harry Potter and **Philosophy**" | "**Harry Potter and The Sorcerer's Stone**" | 5.61 | 🚀 **HUGE WIN** (Fixed intent) |
| **"Jane Austen"** | "A Single Man" (Noise?) | "The Novels of Jane Austen" | 8.96 | ✅ **Precise** |
| **"finding love..."** | "Elusive Love" | "Together Apart" | 6.41 | ✅ **High Quality** |
| **ISBN "0060959479"** | "All About Love" (Rank 1) | "Physical Education..." (Rank 1)<br>"All About Love" (Rank 2) | -1.33 | ⚠️ **Regression** (Model confused by ID) |

## Latency Analysis
- **Cold Start**: ~11s (Model Load).
- **Warm Query**: ~0.7s - 1.5s.
- **Conclusion**: ~1s overhead is acceptable for "Smart Search" mode.

## Optimization Strategy (Next Steps)
1. **Dynamic Reranking**: Only trigger Reranker for natural language queries (detect length > 5 chars or no regex match for ISBN).
2. **Quantization**: Use ONNX version of Cross-Encoder for 2x speedup.
