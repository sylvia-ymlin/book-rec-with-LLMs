# Experiment Archive

This document consolidates all experimental results from the project development.

---

## 1. Retrieval Baseline (2026-01-08)

**System**: ChromaDB (all-MiniLM-L6-v2) - Pure Dense Retrieval

| Query Type | Query | Result | Status |
|:---|:---|:---|:---|
| **Semantic** | "finding love..." | "All About Love" | ✅ SUCCESS |
| **Keyword** | "Harry Potter" | "Harry Potter and Philosophy" | ⚠️ PARTIAL |
| **Exact** | "0060959479" (ISBN) | "National Geographic..." | ❌ FAILURE |

**Conclusion**: Dense retrieval fails on exact entity matching (ISBN). BM25 needed.

---

## 2. Hybrid Search (2026-01-08)

**System**: Hybrid RRF (BM25 + Chroma Dense)

| Query Type | Query | Baseline | Hybrid | Status |
|:---|:---|:---|:---|:---|
| **Semantic** | "finding love..." | "All About Love" | "Elusive Love" | ✅ Maintained |
| **Keyword** | "Harry Potter" | "HP and Philosophy" | **"Sorcerer's Stone"** | 🚀 IMPROVED |
| **Exact** | "0060959479" | National Geographic | **"All About Love"** | 🎉 FIXED |

**Trade-off**: Latency increased from ~20ms (Dense) to ~600ms (Hybrid)

**Tech**: `rank_bm25` (Okapi) + `all-MiniLM-L6-v2` + RRF (k=60)

---

## 3. Cross-Encoder Reranking (2026-01-08)

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

| Query | Hybrid (RRF) | Reranked (Top 1) | Score | Verdict |
|:---|:---|:---|:---|:---|
| "Harry Potter" | HP and Philosophy | **Sorcerer's Stone** | 5.61 | 🚀 HUGE WIN |
| "Jane Austen" | A Single Man | **Novels of Jane Austen** | 8.96 | ✅ Precise |
| "finding love..." | Elusive Love | Together Apart | 6.41 | ✅ High Quality |
| ISBN "0060959479" | All About Love | Physical Education... | -1.33 | ⚠️ Regression |

**Latency**: Cold ~11s, Warm ~0.7-1.5s

**Note**: Reranker confuses ISBNs. Solution: Disable rerank for exact queries.

---

## 4. Agentic Router (2026-01-08)

**Architecture**: Dynamic strategy assignment based on query analysis

| Strategy | Trigger | Pipeline |
|:---|:---|:---|
| **EXACT** | ISBN pattern | BM25 Only |
| **FAST** | Keywords ≤2 words | Hybrid RRF |
| **DEEP** | Complex query | Hybrid + Rerank |

| Query | Strategy | Top Result | Validated? |
|:---|:---|:---|:---|
| "0060959479" | EXACT | "All About Love" | ✅ YES |
| "python programming" | FAST | "Python Cookbook" | ✅ YES |
| "finding love..." | DEEP | "Together Apart" | ✅ YES |

**Impact**: ISBN Precision 100%, Latency optimized per query type.

---

## 5. Temporal Dynamics (2026-01-08)

**Mechanism**: Recency Boosting with Log-Linear Decay

**Formula**: `Score_New = Score_Old + (2.0 / log(Age + e))`

| Title (Year) | Base Score | Temporal Score | Boost | Age |
|:---|:---|:---|:---|:---|
| "Intro to Science" (2011) | 6.076 | **6.772** | +0.696 | 15 yrs |
| "Environmental Sci" (2012) | -0.883 | **-0.173** | +0.710 | 14 yrs |
| "ACP Complete" (1999) | 4.128 | 4.718 | +0.590 | 27 yrs |

**Conclusion**: Successfully implements "Freshness Bias" without burying classics.

---

## 6. Phase 7: YoutubeDNN Integration (2026-01-10)

### Recall Strategy Update

- **Model**: YoutubeDNN (Deep Neural Network)
- **Training**: 50 Epochs, Batch Size 2048, In-Batch Negatives
- **Weighting**: YoutubeDNN (2.0), ItemCF (1.0), UserCF (1.0), Popularity (0.5)

### Deduplication Layer

1. **Context-Aware Filtering**: Exclude user's favorites
2. **Semantic Deduplication**: ISBN → Title mapping for unique results

### Performance Benchmarks

| Metric | Before | After | Improvement |
|:---|:---|:---|:---|
| **Personalized Recs (Cold)** | ~15,000 ms | Pre-warmed | Instant |
| **Personalized Recs (Warm)** | ~50 ms | **19 ms** | 60% faster |
| **Favorites Lookup** | ~100 ms | **84 ms** | 16% faster |
| **Semantic Search** | ~300 ms | **232 ms** | 22% faster |

### Bug Fixes

| Issue | Description | Resolution |
|:---|:---|:---|
| BUG-001 | `ModuleNotFoundError: langchain_openai` | Installed dependency |
| BUG-002 | `TypeError: unhashable type: 'dict'` | Fixed return type |
| BUG-003 | Visual Duplicates | Title-based deduplication |
| BUG-004 | Cold Start Latency | On-Startup model loading |

---

## 7. RecSys Evaluation (2026-01)

### Model Performance (SASRec + XGBoost)

Evaluation: Leave-Last-Out protocol on 500 active users

| Model Configuration | Hit Rate@10 | MRR@5 |
|:---|:---|:---|
| Baseline (ItemCF + Popularity) | 0.4460 | 0.2642 |
| SASRec (3 Epochs) + XGBoost | 0.3660 | 0.1498 |
| SASRec (30 Epochs) + XGBoost | 0.4400 | 0.2089 |

### Feature Importance

**3-Epoch (Undertrained)**:
| Feature | Importance |
|:---|:---|
| sasrec_score | 0.62 (Over-dominant) |
| icf_max | 0.29 |

**30-Epoch (Converged)**:
| Feature | Importance |
|:---|:---|
| icf_max (ItemCF) | 0.60 |
| sasrec_score | 0.26 |
| i_cnt | 0.07 |

**Key Finding**: Undertrained DL features poison traditional ML models.

---

## Data Statistics

| Dataset | Records |
|:---|:---|
| Training Set | 1,079,966 (76.3%) |
| Validation Set | 167,968 (11.9%) |
| Test Set | 167,968 (11.9%) |
| Active Users | 167,968 |
| Books | 221,998 |

---

*Archive Date: January 2026*
