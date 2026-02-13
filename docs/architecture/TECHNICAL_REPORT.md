# Technical Report: Agentic RAG Book Recommender

**Version**: v2.6.0 (Frozen)  
**Author**: [Your Name]  
**Date**: January 2026  
**Project Type**: End-to-End ML/AI System (Retrieval-Augmented Generation + Recommendation System)

---

## Executive Summary

This project implements an integrated Agentic RAG (Retrieval-Augmented Generation) system for book discovery, combined with a personalized recommendation engine. Unlike simple vector search, it uses a self-routing architecture that dynamically selects the optimal retrieval strategy based on query intent.

Key achievements:
- 100% recall on exact-match queries (ISBNs)
- Sub-second latency for keyword searches
- Deep semantic understanding for complex natural language queries
- Detail-level precision via hierarchical (Small-to-Big) retrieval
- Personalized recommendations using 7-channel recall (Item2Vec, Stacking) and LGBMRanker (LambdaRank)

The system demonstrates both Advanced RAG Architecture (Hybrid Search, Reranking, Query Routing) and multi-channel RecSys (Item2Vec, LGBMRanker, Stacking).

---

## 1. Problem Statement

Traditional keyword search fails on modern book discovery scenarios:
- Users search by feeling ("sad sci-fi about AI") rather than keywords
- Users want specific plot details ("books with an unreliable narrator twist")
- Users expect temporal awareness ("latest books on quantum computing")

This system addresses these challenges through:
1. Understanding user intent (Agentic Router)
2. Fusing multiple retrieval strategies (Hybrid Search)
3. Ranking results by semantic relevance (Cross-Encoder Reranking)
4. Finding hidden details in review text (Small-to-Big Retrieval)
5. Personalized recommendations without explicit queries

---

## 2. System Architecture

### 2.1 RAG Pipeline

> 可视图表（流程图、时序图、ER 图）见 [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)。

```
USER QUERY
     |
     v
+---------------------------+
|   AGENTIC QUERY ROUTER    |
|  +------+ +------+ +------+ +------+
|  | ISBN | |Keyword| |Complex| |Detail|
|  |(Exact)| |(Fast) | |(Deep) | |(S2B) |
|  +--+---+ +--+---+ +--+---+ +--+---+
+-----|--------|--------|--------|----+
      |        |        |        |
      v        v        v        v
  +------+ +--------+ +--------+ +--------+
  | BM25 | |Hybrid  | |Hybrid  | |Chunk   |
  | Only | |(RRF)   | |+Rerank | |->Parent|
  +--+---+ +---+----+ +---+----+ +---+----+
     |         |          |          |
     +----+----+----+-----+-----+----+
          |
          v
+---------------------------+
|   POST-PROCESSING         |
|  - Temporal Dynamics      |
|  - Context Compression    |
+---------------------------+
          |
          v
+---------------------------+
|   LLM GENERATION          |
|   (Streaming via SSE)     |
+---------------------------+
```

### 2.2 Recommendation Pipeline

```
USER REQUEST (No Query)
          |
          v
+---------------------------+
|   7-CHANNEL RECALL (RRF)  |
|  - ItemCF (direction wt)  |
|  - UserCF (Jaccard)       |
|  - Swing (user-pair)      |
|  - SASRec (embedding)     |
|  - Item2Vec (Word2Vec)    |
|  - YoutubeDNN (two-tower) |
|  - Popularity (fallback)  |
+---------------------------+
          |
          v
+---------------------------+
|   FEATURE ENGINEERING     |
|  - User / Item stats      |
|  - SASRec score           |
|  - ItemCF / UserCF scores |
|  - Author / Category aff  |
+---------------------------+
          |
          v
+---------------------------+
|   LGBMRanker (LambdaRank) |
|   Optimizes NDCG directly |
+---------------------------+
          |
          v
      TOP-K RESULTS
```

---

## 3. Technical Components

### 3.1 Agentic Query Router

Location: `src/rag/router.py`

Rule-based intent classifier using RegEx and keyword detection:

| Query Type | Detection Logic | Strategy | Latency |
|------------|-----------------|----------|---------|
| ISBN | `\d{10,13}` pattern | BM25 Only (alpha=1.0) | <100ms |
| Keyword | `len(words) <= 2` | Hybrid (No Rerank) | ~300ms |
| Complex | Default | Hybrid + Cross-Encoder | ~800ms |
| Detail | Keywords: "twist", "ending" | Small-to-Big | ~500ms |

Trade-off: Rule-based routing was chosen over LLM-based routing to avoid 500ms+ latency per routing decision.

### 3.2 Hybrid Search with RRF

Location: `src/rag/vector_db.py`

Combines sparse retrieval (BM25) and dense retrieval (ChromaDB/MiniLM) using Reciprocal Rank Fusion:

```
RRF_Score = sum(1/(k + rank_dense) + 1/(k + rank_sparse))  where k=60
```

Result: 100% recall on ISBNs (previously 0% with pure vector search).

### 3.3 Cross-Encoder Reranking

Location: `src/rag/reranker.py`

Two-stage retrieval:
1. Stage 1: Retrieve top-50 candidates via RRF (~100ms)
2. Stage 2: Rerank with `ms-marco-MiniLM-L-6-v2` (~400ms)

Trade-off: Only rerank top-50 (not all 200K) to balance precision vs latency.

### 3.4 Small-to-Big Retrieval

Location: `src/rag/vector_db.py::small_to_big_search`

Implementation (based on LlamaIndex Parent-Child, RAPTOR):
1. Chunking: 788,174 review sentences indexed at sentence-level
2. Matching: Query matches specific sentence ("I cried at the ending")
3. Expansion: Map sentence to parent ISBN to full book context

Result: Can answer queries like "books with unreliable narrator twist" that are invisible to description-level search. *RAG components (ISBN recall, reranking, Small-to-Big) were validated via curated examples and routing statistics; no large-scale human evaluation was conducted.*

### 3.5 Temporal Dynamics

Location: `src/rag/temporal.py`

Log-linear decay function to boost newer documents:

```
Score_new = Score_old + 2.0 / ln(Age + 2.718)
```

Triggered by keywords: "new", "latest", "2024".

### 3.6 Context Compression

Location: `src/rag/context_compressor.py`

- Retains last 2 turns (4 messages) raw
- Summarizes older turns using lightweight LLM call
- Enables infinite conversation without token overflow

---

## 4. Personalized Recommendation System

### 4.1 Multi-Channel Recall (7 Channels)

| Recall Channel | Algorithm | Weight | Purpose |
|:---|:---|:---|:---|
| ItemCF | Co-rating similarity with direction weight (forward=1.0, backward=0.7) | 1.0 | Collaborative filtering |
| UserCF | User similarity (Jaccard + activity penalty) | 1.0 | Similar user preferences |
| Swing | User-pair overlap weighting: `1/(α + \|I_u ∩ I_v\|)` | 1.0 | Substitute relationships |
| SASRec | Dot-product retrieval from pre-computed embeddings | 1.0 | Sequential patterns |
| Item2Vec | Word2Vec (Skip-gram) on user interaction sequences | 0.8 | Implicit co-occurrence |
| YoutubeDNN | Two-tower user-item dot product | 0.1 | Deep learning recall |
| Popularity | Rating count with time decay | 0.5 | Cold-start fallback |

Fusion: Reciprocal Rank Fusion — `score += weight * (1 / (k + rank + 1))`, k=60

ItemCF formula:
```
loc_alpha = 1.0 if item1 before item2 else 0.7  # direction weight
loc_weight = loc_alpha * (0.9 ^ (|loc1 - loc2| - 1))
time_weight = 1 / (1 + 10 * |t1 - t2|)
rating_weight = (r1 + r2) / 10
sim[i][j] = sum(loc * time * rating * user_penalty) / sqrt(cnt[i] * cnt[j])
```

### 4.2 SASRec Sequential Model

Architecture: Self-Attentive Sequential Recommendation with Transformer blocks
- Training: 30 epochs, 64-dim embeddings, BCE loss with negative sampling
- Dual use: (1) ranking feature via `sasrec_score`, (2) independent recall channel via embedding dot-product

**Time-split (no leakage)**: SASRec is trained on `train.csv` only. `user_seq_emb` and `sas_item_emb` are computed from train-only sequences. When Ranking uses `sasrec_score` for val samples, the user's history contains only train interactions—never val/test. `build_sequences.py` and SASRec/YoutubeDNN all use train-only.

### 4.3 LGBMRanker (LambdaRank) + Model Stacking

Replaced XGBoost binary classifier with LightGBM LambdaRank that directly optimizes NDCG. In v2.6.0, a Stacking ensemble (LGBMRanker + XGBClassifier → LogisticRegression meta-learner) further improves ranking robustness.

**Training strategy**:
- Hard negative sampling: negatives mined from recall results (not random items)
- 20K users sampled from 168K validation set for training speed
- 4× negative ratio per positive sample

**Feature consistency**: Recall models (SASRec, ItemCF, etc.) are trained on train.csv. Ranking labels come from val.csv. Features like `sasrec_score` use train-only embeddings. Pipeline order: `split_rec_data` → `build_sequences` (train-only) → recall models (train) → ranker (val).

**17 features** in 5 groups:
- User statistics: u_cnt, u_mean, u_std
- Item statistics: i_cnt, i_mean, i_std
- Cross features: len_diff, u_auth_avg, u_auth_match, is_cat_hob
- Sequence: sasrec_score, sim_max, sim_min, sim_mean
- CF scores: icf_sum, icf_max, ucf_sum

Feature importance (v2.6.0 LGBMRanker, representative subset):

| Feature | Importance | Description |
|:---|:---|:---|
| u_cnt | 88 | User activity count |
| sim_max | 76 | Last-N similarity max |
| icf_max | 62 | ItemCF max similarity |
| i_cnt | 59 | Item popularity count |
| len_diff | 55 | Description complexity match |
| sasrec_score | 25 | SASRec embedding score |

### 4.4 Evaluation Results

*Protocol: Leave-Last-Out, n=2000 users, title-relaxed matching, filter_favorites=False.*

| Metric | V2.0 (XGBoost) | V2.5 (LGBMRanker) | v2.6.0 (+Item2Vec, Stacking) |
|:---|:---|:---|:---|
| HR@10 | 0.1380 | 0.2205 | **0.4545** |
| MRR@5 | 0.1295 | 0.1584 | **0.2893** |
| Dataset | 167,968 active users, 221,998 books | | |

---

## 5. Performance Metrics

### 5.1 RAG System

| Metric | Baseline (Vector Only) | This System |
|--------|------------------------|-------------|
| ISBN Recall | 0% | 100% |
| Keyword Precision | Low | High (BM25 boost) |
| Detail Query Recall | 0% | Golden Test Set (Accuracy@K, Recall@K, MRR@K, NDCG@K) |
| Avg Latency | 100ms | 300-800ms |
| Chat Context Limit | ~10 turns | Extended via compression (no formal limit) |

**Golden Test Set**: Human-annotated Query-Book pairs (`data/rag_golden.csv`) for quantitative RAG evaluation. Run `python scripts/model/evaluate_rag.py` for Accuracy@K, Recall@K, MRR@K, NDCG@K. Extend with ~500+ pairs for production.

### 5.2 Latency Benchmarks

| Operation | P50 Latency (Warm) | P95 Latency (Warm) |
|:---|:---|:---|
| Exact ISBN Search | 19 ms | 45 ms |
| Semantic Search (Hybrid) | 232 ms | 310 ms |
| Search + Reranking (Top 50) | 710 ms | 1,250 ms |

### 5.3 Reranking Impact

| Query | Hybrid (Raw RRF) | Reranked (Rank 1) | Verdict |
|:---|:---|:---|:---|
| "Harry Potter" | Harry Potter and Philosophy | The Sorcerer's Stone | Corrected Intent |
| "Jane Austen" | A Single Man | Novels of Jane Austen | Noise Reduction |

---

## 6. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Vector DB | ChromaDB | Embedded, zero-latency vector storage |
| Sparse Index | BM25Okapi (rank_bm25) | Keyword/exact match retrieval |
| Embeddings | all-MiniLM-L6-v2 | 384-dim sentence embeddings |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder precision ranking |
| LLM | OpenAI / Ollama (llama3) | Generation with BYOK support |
| Backend | FastAPI + SSE | Streaming API |
| Frontend | React 18 + Vite | Modern SPA |
| Ranking | LightGBM (LambdaRank) | List-wise NDCG optimization |
| Sequential | SASRec (PyTorch) | Transformer-based sequence modeling |

---

## 7. Design Decisions

| Decision | Chosen Option | Rejected Alternative | Rationale |
|----------|---------------|---------------------|-----------|
| Vector DB | ChromaDB (embedded) | Pinecone (cloud) | Zero network latency; 200K docs fits in RAM |
| Routing | Rule-based RegEx | LLM-based routing | 2ms vs 500ms latency; deterministic behavior |
| Reranking | Cross-Encoder | LLM reranking | 400ms vs 2s latency; proven accuracy |
| Chunking | Sentence-level (Small-to-Big) | Fixed 512 tokens | Semantic integrity; detail-level matching |
| SFT Data | Self-Instruct | Manual annotation | Scalable; leverages existing reviews |
| Freshness fallback writes | Staging store (`online_books.db`) | Append to `books_processed.csv` | Data: training CSV stays frozen. perf: main `books.db` read-only; no write lock contention |

### 7.1 Staging Store for Online Writes

When `freshness_fallback` fetches books from Google Books, they are written to a **separate** `online_books.db` SQLite file instead of the main store. This decouples:

1. **Data risk**: `books_processed.csv` and `books.db` remain frozen for training; no distribution shift.
2. **Performance**: Main `books.db` is read-only during serving; writes go only to `online_books.db`, avoiding lock contention on high-concurrency reads.

Lookup: `metadata_store.get_book_metadata()` checks main first, then `online_books_store`. FTS5 search merges results from both indices.

---

## 8. SFT Data Pipeline (Supplementary)

*Not integrated into the main RAG flow in v2.6.0.* This pipeline was developed for potential future fine-tuning of chat tone.

### 8.1 Problem

Default LLM tone is corporate; the target is "Literary Critic" personality.

### 8.2 Implementation (Self-Instruct with LLM-as-a-Judge)

Pipeline:
1. Seed Selection: Sample 1000 high-emotion reviews (rating=5, length>200)
2. Instruction Evolution: GPT generates user questions that would prompt each review
3. Response Transform: Rewrite reviews as AI assistant style
4. LLM-as-a-Judge: Filter for Empathy/Specificity/Critique Depth >= 8/10

Output:
- `data/sft/literary_critic_train.jsonl`: ~800 high-quality (Query, Response) pairs
- `data/dpo/preference_pairs.jsonl`: ~500 (Chosen, Rejected) pairs

See [Experiment Archive](experiments/experiment_archive.md) for full implementation details.

---

## 9. File Structure

```
src/
├── app/
│   ├── main.py                # FastAPI entrypoint
│   └── api/chat.py            # Chat router
├── rag/
│   ├── router.py              # Agentic Query Router
│   ├── vector_db.py           # Hybrid Search + Small-to-Big
│   ├── reranker.py            # Cross-Encoder Reranking
│   ├── temporal.py            # Recency Boosting
│   └── context_compressor.py  # Chat History Compression
├── recsys/
│   ├── recall/                # 7-channel recall
│   └── ranking/               # Ranking modules
├── data/
│   ├── repository.py          # Unified data access
│   └── stores/                # metadata / profile / online books
├── core/
│   ├── diversity_reranker.py  # P0: MMR + popularity penalty + category constraint
│   └── diversity_metrics.py   # P3: Category Coverage, ILSD
├── support/
│   ├── data_factory/          # SFT Data Synthesis + LLM Judge
│   └── marketing/             # Persona/highlights
├── services/
│   ├── chat_service.py        # RAG Chat Pipeline
│   └── recommend_service.py   # Personalized Recommendation
└── model/sasrec.py            # SASRec model definition
```

---

## 9.1 P0–P3 Optimizations (Post-v2.6)

| Priority | Optimization | Location | Description |
|:---|:---|:---|:---|
| **P0** | Diversity Rerank | `DiversityReranker`, `RecommendationService` | MMR (λ=0.75), popularity penalty, max 3 per category in top-k |
| **P1** | Real-time Sequence | `SASRecRecall`, `DINRanker`, `FeatureEngineer`, `RecommendationService` | `real_time_sequence` merges session ISBNs into recall/ranking |
| **P2** | Hard/Random Ratio | `train_ranker.py`, `train_din_ranker.py` | `--hard_ratio 0.5` for half hard half random negatives |
| **P3** | Diversity Metrics | `evaluate.py`, `diversity_metrics.py` | Category Coverage@10, ILSD@10 reported |
| **P3** | Hard Neg Filter | `train_ranker.py --filter_similar` | Exclude hard negs with embedding sim > 0.9 to positive |

---

## 10. Limitations

- **Single-dataset evaluation**: All RecSys metrics are on Amazon Books 200K; no cross-domain or external validation.
- **Rule-based router**: Intent classification uses heuristics (e.g., `len(words) <= 2` for keyword); may not generalize to other domains.
- **RAG evaluation**: Use Golden Test Set (`data/rag_golden.csv`) for Accuracy@K, Recall@K, MRR@K. Extend to 500+ human-annotated Query-Book pairs for production.
- **Protocol sensitivity**: RecSys metrics can vary with evaluation protocol (e.g., ISBN-only vs title-relaxed matching); see [Experiment Archive](experiments/experiment_archive.md) for discussion.

---

## 11. Scalability

Current capacity:
- In-memory index: 2GB RAM, ~200K books
- Vertical scaling: Can handle ~5M books on standard server

Migration path:
- Horizontal scaling: Qdrant/Pinecone for >10k concurrent users
- Distributed training: Ray for large-scale model training

---

## References

1. Self-Instruct (Wang et al., 2022) - Instruction data synthesis
2. RAPTOR (Sarthi et al., 2024) - Hierarchical tree-based indexing
3. HyDE (Gao et al., 2022) - Hypothetical document embeddings
4. LlamaIndex - Parent-child retrieval patterns
5. ms-marco-MiniLM - Cross-encoder reranking
6. SASRec (Kang et al., 2018) - Self-Attentive Sequential Recommendation
