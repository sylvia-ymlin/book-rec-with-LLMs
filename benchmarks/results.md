# Performance Benchmark Results

**Date**: 2026-01-06
**Environment**: Hugging Face Spaces (Production)

## System Info

- Dataset: 5,000+ books
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- Vector Database: ChromaDB with HNSW index
- Deployment: Docker container on Hugging Face Spaces

## Results (Live Production Test)

| Metric | Value |
|--------|-------|
| **Backend Processing Time** | 0.3 - 0.4 seconds |
| **End-to-End Response** | < 1 second |
| **Dataset Size** | 5,000+ books |
| **Test Query** | "a philosophical novel about the meaning of life" |

### Sample Results Returned
1. *Discover Your Destiny with the Monk Who Sold His Ferrari* by Robin Sharma
2. *The Devil and Miss Prym* by Paulo Coelho
3. *The Perennial Philosophy* by Aldous Huxley
4. *Zen and the Art of Motorcycle Maintenance* by Robert M. Pirsig

## Performance Breakdown (Estimated)

| Component | Latency |
|-----------|---------|
| Vector Search (ChromaDB/HNSW) | ~100-200ms |
| Category/Emotion Filtering | ~5-10ms |
| Result Formatting | ~5ms |
| Network Round-trip | ~100-200ms |
| **Total** | **~300-400ms** |

## Notes

- The 0.3-0.4s backend time is displayed by Gradio's internal timer
- End-to-end response includes network latency and UI rendering
- Performance is consistent across multiple queries
- No errors or timeouts observed during testing

## Test Evidence

Browser test recording available at: `hf_benchmark_test_*.webp`
