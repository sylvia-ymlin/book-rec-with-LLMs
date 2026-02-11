# Retrieval Baseline Report
**Date**: 2026-01-08
**Metric**: Recall@5 (Qualitative check)

## Experiment Setup
- **System**: ChromaDB (all-MiniLM-L6-v2) - Pure Dense Retrieval.
- **Dataset**: Book Reviews (~220k docs).
- **Benchmarks**:
    1.  **Semantic Queries** (e.g., "finding love"): Expected STRONG performance.
    2.  **Keyword Queries** (e.g., "Harry Potter"): Expected MODERATE performance.
    3.  **Exact Match** (e.g., ISBN): Expected WEAK performance.

## Results
| Query Type | Query | Result | Status |
| :--- | :--- | :--- | :--- |
| **Semantic** | "finding love..." | "All About Love" (found via similar vector) | ✅ **SUCCESS** |
| **Keyword** | "Harry Potter" | "Harry Potter and Philosophy" | ⚠️ **PARTIAL** (Found related, but missed main novels?) |
| **Exact** | "0060959479" | "National Geographic..." (Completely unrelated) | ❌ **FAILURE** |

## Analysis
The current **Dense Retrieval** model treats the ISBN `0060959479` as a semantic string. Since the embedding model (MiniLM) is not trained to recognize ISBN relationships, it maps the number to a vector space location that happens to be near "National Geographic" (likely random noise collision or digit similarity).

**Conclusion**: The system is **incapable of exact entity retrieval** by ID or specific unique identifier.

## Optimization Plan
**Implement Hybrid Search** to combine:
1.  **BM25 (Sparse)**: For exact keyword/ID matching.
2.  **Vector (Dense)**: For semantic understanding.
