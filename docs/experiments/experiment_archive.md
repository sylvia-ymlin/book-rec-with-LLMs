# Experiment Archive

This document consolidates all experimental results from the project development.

> **Frozen at v2.6.0** — Experiments recorded as of January 2026. No new experiments.

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

## 8. V2.5 RecSys Enhancements (2026-01-29)

### Problem

After the performance debugging in Section 7, the system sat at HR@10=0.1380 / MRR@5=0.1295 (n=500). Two structural problems remained:

1. **ItemCF direction weight not applied** — `build_recall_models.py` had `if itemcf.load(): skip` logic, so the new asymmetric similarity (forward=1.0, backward=0.7) never took effect. The on-disk `itemcf.pkl` was stale.
2. **Swing recall too slow to train** — The original implementation iterated `items → shared_users → user_pairs`, which is O(items × users²). On 133K items / 1M+ interactions, it only processed 773/133816 items in 46 seconds (~2-3 hours estimated). Training was killed.
3. **No SASRec recall channel** — SASRec was only used as a ranking feature (`sasrec_score`), not as an independent recall source.
4. **XGBoost optimized AUC, not NDCG** — Binary classification loss doesn't directly optimize list-wise ranking quality.
5. **Random negative sampling** — Ranker was trained against random items, not against "close but wrong" candidates from recall.

### Changes Implemented

#### Recall Layer

| Change | Detail |
|:---|:---|
| **ItemCF direction weight** | `loc_alpha = 1.0 if loc1 < loc2 else 0.7` — biases `sim[earlier][later] > sim[later][earlier]` |
| **Forced retrain** | Removed `if itemcf.load(): skip` so the direction weight change actually applies |
| **Swing (optimized)** | Rewrote algorithm: iterate `users → item_pairs` instead of `items → users → pairs`. Complexity drops from O(items × users²) to O(users × items_per_user²). Added `max_hist=50` cap per user. |
| **SASRec recall channel** | New `src/recsys/recall/sasrec_recall.py` — loads pre-computed `user_seq_emb.pkl` + `item_emb.weight` from model checkpoint, does dot-product retrieval |

Recall channel weights after V2.5:

| Channel | Weight |
|:---|:---|
| YoutubeDNN | 0.1 |
| ItemCF | 1.0 |
| UserCF | 1.0 |
| Swing | 1.0 |
| SASRec | 1.0 |
| Popularity | 0.5 |

#### Ranking Model

| Change | Detail |
|:---|:---|
| **XGBoost → LGBMRanker** | `objective='lambdarank'`, `metric='ndcg'`, optimizes list-wise ranking directly |
| **Hard negative sampling** | Negatives mined from recall results (items recalled but not the positive) instead of random items |
| **Sampling for speed** | 20K users sampled from 168K val set — sufficient for LTR, reduces mining time from ~1.5h to ~16 min |

### Training Time (CPU, Apple Silicon)

| Model | Time | Notes |
|:---|:---|:---|
| ItemCF | 2 min 6 sec | Full retrain with direction weight |
| UserCF | 7 sec | |
| **Swing** | **35 sec** | Was ~2-3 hours before optimization |
| Popularity | <1 sec | |
| LGBMRanker | ~16 min | 20K users × 4 hard negatives, 17 features |

### Swing Algorithm Optimization Detail

**Before** (killed after 46 sec, 773/133816 items):
```
for item_i in all_items:           # 133K
    for user in users_of(item_i):   # variable
        for item_j in items_of(user):  # variable
            pair_users[(i,j)].append(user)
            for u2 in pair_users[(i,j)]:  # O(n²) user-pair
                score += 1/(alpha + overlap(u, u2))
```

**After** (35 sec total):
```
# Phase 1: iterate users, enumerate item pairs
for user in all_users:              # 168K
    items = user_items[user][:50]   # capped
    for i, j in combinations(items):
        pair_users[(i,j)].append(user)

# Phase 2: compute swing per item pair
for (i,j), users in pair_users:     # 5.28M pairs
    for u, v in combinations(users[:100]):
        score += 1/(alpha + overlap(u,v))
```

Key optimizations:
- User-centric iteration instead of item-centric (exploits sparsity)
- `max_hist=50` caps user history (removes noisy power users)
- `users[:100]` caps user-pair computation per item pair
- Canonical `(i,j)` ordering avoids duplicate pairs

### Feature Importance (LGBMRanker, 17 features)

| Feature | Importance | Description |
|:---|:---|:---|
| i_cnt | 96 | Item popularity count |
| sim_max | 91 | Last-N similarity max |
| u_cnt | 80 | User activity count |
| i_mean | 41 | Item average rating |
| len_diff | 28 | Description complexity match |
| icf_max | 23 | ItemCF max similarity |
| sasrec_score | 22 | SASRec embedding score |
| icf_sum | 21 | ItemCF sum similarity |
| i_std | 20 | Item rating std dev |
| u_mean | 17 | User average rating |
| sim_mean | 17 | Last-N similarity mean |
| sim_min | 15 | Last-N similarity min |
| u_std | 9 | User rating std dev |
| ucf_sum | 9 | UserCF sum similarity |
| u_auth_avg | 2 | User-author affinity |
| u_auth_match | 0 | Author match flag |
| is_cat_hob | 0 | Category hobby match |

**Key shift**: `i_cnt` (96) and `sim_max` (91) now dominate over `icf_max` (23). Previously in XGBoost, `icf_max` was 0.60. This suggests the LGBMRanker relies more on popularity and sequence similarity signals, while ItemCF is still useful but less dominant.

### Results

Evaluation: Leave-Last-Out protocol, title-relaxed matching, `filter_favorites=False`

| Configuration | HR@10 | MRR@5 | Sample |
|:---|:---|:---|:---|
| Post-debugging baseline | 0.1380 | 0.1295 | n=500 |
| **V2.5 (full pipeline)** | **0.1940** | **0.1419** | n=500 |
| **V2.5 (full pipeline)** | **0.2205** | **0.1584** | n=2000 |

**Relative improvement** (n=2000 vs baseline):
- HR@10: **+59.8%** (0.1380 → 0.2205)
- MRR@5: **+22.3%** (0.1295 → 0.1584)

### Gap to Original Baseline

The original ItemCF+Popularity baseline (Section 7) scored HR@10=0.4460. The V2.5 system at 0.2205 is still below that number. Possible reasons:

1. **Evaluation protocol difference** — the original baseline was tested under strict ISBN-only matching on a different sample; V2.5 uses title-relaxed matching + `filter_favorites=False` which changes the comparison.
2. **YoutubeDNN weight (0.1) may still inject noise** — even at low weight, poor recall candidates enter the fusion pool.
3. **SASRec recall channel** may not be loading correctly if the pre-computed embeddings are outdated.
4. **Title deduplication** removes valid candidates when different editions exist.

### Next Steps

- Re-evaluate the original baseline under the same evaluation protocol (title-relaxed, `filter_favorites=False`) for fair comparison
- Experiment with disabling YoutubeDNN entirely
- Verify SASRec recall is returning meaningful candidates
- Consider increasing `neg_ratio` or `max_samples` for ranker training

---

## 9. V2.6 Item2Vec + Model Stacking (2026-01-29)

### Problem

V2.5 achieved HR@10=0.2205 / MRR@5=0.1584 (n=2000). Two P2 backlog items remained:

1. **No embedding-based recall from interaction sequences** — SASRec provided sequence embeddings, but no simpler Word2Vec-based approach existed to capture implicit item co-occurrence patterns.
2. **Single ranking model** — LGBMRanker alone, with no ensemble diversification to reduce overfitting to a single model's biases.

### Changes Implemented

#### Recall Layer: Item2Vec

| Aspect | Detail |
|:---|:---|
| **Algorithm** | Word2Vec (Skip-gram) on user interaction sequences |
| **Reference** | Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for Collaborative Filtering", 2016 |
| **Parameters** | `vector_size=64, window=5, min_count=3, sg=1, epochs=10, workers=4` |
| **Vocabulary** | 44,157 items (from 133K+ total; rest below min_count threshold) |
| **Similarity matrix** | Top-200 most similar items per vocabulary item (cosine similarity) |
| **Fusion weight** | 0.8 (between Popularity 0.5 and CF channels 1.0) |
| **Training time** | ~48 seconds (index build 15s + Word2Vec 7s + similarity matrix 22s) |

Implementation: `src/recsys/recall/item2vec.py` — follows Swing/ItemCF interface pattern exactly (`__init__`, `fit`, `recommend`, `save`, `load`).

#### Ranking Model: Model Stacking

| Aspect | Detail |
|:---|:---|
| **Architecture** | Level-1: LGBMRanker + XGBClassifier → Level-2: LogisticRegression |
| **CV Strategy** | 5-Fold GroupKFold (preserves user query groups) |
| **Level-1A** | LGBMRanker: `lambdarank`, n_estimators=100, max_depth=6 |
| **Level-1B** | XGBClassifier: `binary:logistic`, n_estimators=100, max_depth=6 |
| **Level-2** | LogisticRegression: `solver='lbfgs'`, max_iter=1000, C=1.0 |
| **Training** | OOF predictions from CV → Meta-learner, then full retrain Level-1 for inference |

**Meta-learner coefficients**: `LGB=1.4901` (dominant), `XGB=0.0420` (small positive contribution), `intercept=-0.1171`

The LGB coefficient is ~35× larger than XGB, indicating LGBMRanker's LambdaRank scores carry most of the ranking signal. XGB still provides a small but positive contribution, confirming the value of ensemble diversity.

### Recall Channel Weights (V2.6, 7 channels)

| Channel | Weight | New? |
|:---|:---|:---|
| YoutubeDNN | 0.1 | |
| ItemCF | 1.0 | |
| UserCF | 1.0 | |
| Swing | 1.0 | |
| SASRec | 1.0 | |
| **Item2Vec** | **0.8** | ✅ New |
| Popularity | 0.5 | |

### Feature Importance (LGBMRanker, full retrained, 17 features)

| Feature | Importance | Description |
|:---|:---|:---|
| u_cnt | 88 | User activity count |
| sim_max | 76 | Last-N similarity max |
| icf_max | 62 | ItemCF max similarity |
| i_cnt | 59 | Item popularity count |
| len_diff | 55 | Description complexity match |
| sim_mean | 48 | Last-N similarity mean |
| i_mean | 47 | Item average rating |
| i_std | 43 | Item rating std dev |
| ucf_sum | 38 | UserCF sum similarity |
| icf_sum | 33 | ItemCF sum similarity |
| sim_min | 32 | Last-N similarity min |
| sasrec_score | 25 | SASRec embedding score |
| u_mean | 24 | User average rating |
| u_std | 15 | User rating std dev |
| u_auth_avg | 7 | User-author affinity |
| u_auth_match | 1 | Author match flag |
| is_cat_hob | 0 | Category hobby match |

**Key shift from V2.5**: `u_cnt` (88) overtook `i_cnt` (96→59) as the top feature. `icf_max` rose from 23 to 62, suggesting Item2Vec's added recall diversity improved the quality of ItemCF similarity signals reaching the ranker.

### Training Time (CPU, Apple Silicon)

| Model | Time | Notes |
|:---|:---|:---|
| **Item2Vec** | **48 sec** | Word2Vec + similarity matrix |
| Hard Negative Mining | ~17 min | 20K users × 4 negatives, 7-channel recall |
| Feature Generation | ~5 sec | 17 features |
| 5-Fold CV + Retrain | <1 sec | LGB + XGB + Meta-Learner |

### Results

Evaluation: Leave-Last-Out protocol, title-relaxed matching, `filter_favorites=False`

| Configuration | HR@10 | MRR@5 | Sample |
|:---|:---|:---|:---|
| V2.5 baseline | 0.2205 | 0.1584 | n=2000 |
| **V2.6 (Item2Vec + Stacking)** | **0.4545** | **0.2893** | **n=2000** |

**Relative improvement** (V2.5 → V2.6):
- HR@10: **+106.1%** (0.2205 → 0.4545)
- MRR@5: **+82.6%** (0.1584 → 0.2893)

### Analysis

The dramatic improvement (+106% HR@10) is likely attributable to:

1. **Item2Vec added recall diversity** — Word2Vec captures implicit co-occurrence patterns that CF methods miss. Items that are semantically similar in embedding space but don't share explicit co-ratings can now be recalled.
2. **Stacking reduced ranking errors** — While LGB dominates (coeff 1.49 vs 0.04), XGB's binary classification perspective provides a complementary signal that catches cases where LambdaRank scores are misleading.
3. **7-channel recall breadth** — More diverse candidates entering the ranker means more "correct" items have a chance to be ranked highly.
4. **Hard negative quality improved** — With 7 recall channels, hard negatives are more challenging and informative, improving ranker discrimination.

### Files Changed

| File | Action |
|:---|:---|
| `src/recsys/recall/item2vec.py` | **New** — Item2Vec recall model |
| `src/recsys/recall/fusion.py` | Modified — added 7th recall channel |
| `scripts/model/build_recall_models.py` | Modified — added Item2Vec training |
| `scripts/model/train_ranker.py` | Modified — added `train_stacking()` + CLI |
| `src/services/recommend_service.py` | Modified — stacking inference with backward compatibility |
| `config/data_config.py` | Modified — 3 new path constants |
| `requirements.txt` | Modified — added gensim, xgboost |

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

*Frozen January 2026 — v2.6.0*
