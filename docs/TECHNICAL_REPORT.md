# Technical Report: Agentic RAG Book Recommender

**Version**: v2.6.0 (Frozen)  
**Author**: [Your Name]  
**Date**: January 2026  
**Project Type**: End-to-End ML/AI System (Retrieval-Augmented Generation + Recommendation System)

---

## Executive Summary

This project implements a production-grade Agentic RAG (Retrieval-Augmented Generation) system for book discovery, combined with a personalized recommendation engine. Unlike simple vector search, it uses a self-routing architecture that dynamically selects the optimal retrieval strategy based on query intent.

Key achievements:
- 100% recall on exact-match queries (ISBNs)
- Sub-second latency for keyword searches
- Deep semantic understanding for complex natural language queries
- Detail-level precision via hierarchical (Small-to-Big) retrieval
- Personalized recommendations using 7-channel recall (Item2Vec, Stacking) and LGBMRanker (LambdaRank)

The system demonstrates mastery of both Data-Centric AI (SFT data synthesis) and Advanced RAG Architecture (Hybrid Search, Reranking, Query Routing).

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
|   6-CHANNEL RECALL (RRF)  |
|  - ItemCF (direction wt)  |
|  - UserCF (Jaccard)       |
|  - Swing (user-pair)      |
|  - SASRec (embedding)     |
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

Location: `src/core/router.py`

Rule-based intent classifier using RegEx and keyword detection:

| Query Type | Detection Logic | Strategy | Latency |
|------------|-----------------|----------|---------|
| ISBN | `\d{10,13}` pattern | BM25 Only (alpha=1.0) | <100ms |
| Keyword | `len(words) <= 2` | Hybrid (No Rerank) | ~300ms |
| Complex | Default | Hybrid + Cross-Encoder | ~800ms |
| Detail | Keywords: "twist", "ending" | Small-to-Big | ~500ms |

Trade-off: Rule-based routing was chosen over LLM-based routing to avoid 500ms+ latency per routing decision.

### 3.2 Hybrid Search with RRF

Location: `src/vector_db.py`

Combines sparse retrieval (BM25) and dense retrieval (ChromaDB/MiniLM) using Reciprocal Rank Fusion:

```
RRF_Score = sum(1/(k + rank_dense) + 1/(k + rank_sparse))  where k=60
```

Result: 100% recall on ISBNs (previously 0% with pure vector search).

### 3.3 Cross-Encoder Reranking

Location: `src/core/reranker.py`

Two-stage retrieval:
1. Stage 1: Retrieve top-50 candidates via RRF (~100ms)
2. Stage 2: Rerank with `ms-marco-MiniLM-L-6-v2` (~400ms)

Trade-off: Only rerank top-50 (not all 200K) to balance precision vs latency.

### 3.4 Small-to-Big Retrieval

Location: `src/vector_db.py::small_to_big_search`

Implementation (based on LlamaIndex Parent-Child, RAPTOR):
1. Chunking: 788,174 review sentences indexed at sentence-level
2. Matching: Query matches specific sentence ("I cried at the ending")
3. Expansion: Map sentence to parent ISBN to full book context

Result: Can answer queries like "books with unreliable narrator twist" that are invisible to description-level search.

### 3.5 Temporal Dynamics

Location: `src/core/temporal.py`

Log-linear decay function to boost newer documents:

```
Score_new = Score_old + 2.0 / ln(Age + 2.718)
```

Triggered by keywords: "new", "latest", "2024".

### 3.6 Context Compression

Location: `src/core/context_compressor.py`

- Retains last 2 turns (4 messages) raw
- Summarizes older turns using lightweight LLM call
- Enables infinite conversation without token overflow

---

## 4. Personalized Recommendation System

### 4.1 Multi-Channel Recall (6 Channels)

| Recall Channel | Algorithm | Weight | Purpose |
|:---|:---|:---|:---|
| ItemCF | Co-rating similarity with direction weight (forward=1.0, backward=0.7) | 1.0 | Collaborative filtering |
| UserCF | User similarity (Jaccard + activity penalty) | 1.0 | Similar user preferences |
| Swing | User-pair overlap weighting: `1/(α + \|I_u ∩ I_v\|)` | 1.0 | Substitute relationships |
| SASRec | Dot-product retrieval from pre-computed embeddings | 1.0 | Sequential patterns |
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

### 4.3 LGBMRanker (LambdaRank)

Replaced XGBoost binary classifier with LightGBM LambdaRank that directly optimizes NDCG.

**Training strategy**:
- Hard negative sampling: negatives mined from recall results (not random items)
- 20K users sampled from 168K validation set for training speed
- 4× negative ratio per positive sample

**17 features** in 5 groups:
- User statistics: u_cnt, u_mean, u_std
- Item statistics: i_cnt, i_mean, i_std
- Cross features: len_diff, u_auth_avg, u_auth_match, is_cat_hob
- Sequence: sasrec_score, sim_max, sim_min, sim_mean
- CF scores: icf_sum, icf_max, ucf_sum

Feature importance (V2.5 LGBMRanker):

| Feature | Importance | Description |
|:---|:---|:---|
| i_cnt | 96 | Item popularity count |
| sim_max | 91 | Last-N similarity max |
| u_cnt | 80 | User activity count |
| i_mean | 41 | Item average rating |
| sasrec_score | 22 | SASRec embedding score |
| icf_max | 23 | ItemCF max similarity |

### 4.4 Evaluation Results

| Metric | V2.0 (XGBoost) | V2.5 (LGBMRanker) | Improvement |
|:---|:---|:---|:---|
| HR@10 | 0.1380 | **0.2205** | +59.8% |
| MRR@5 | 0.1295 | **0.1584** | +22.3% |
| Users Evaluated | 500 | 2,000 | |
| Dataset | 167,968 active users, 221,998 books | | |

---

## 5. Performance Metrics

### 5.1 RAG System

| Metric | Baseline (Vector Only) | This System |
|--------|------------------------|-------------|
| ISBN Recall | 0% | 100% |
| Keyword Precision | Low | High (BM25 boost) |
| Detail Query Recall | 0% | High (Small-to-Big) |
| Avg Latency | 100ms | 300-800ms |
| Chat Context Limit | ~10 turns | Unlimited (compression) |

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

---

## 8. SFT Data Pipeline

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

---

## 9. File Structure

```
src/
├── core/
│   ├── router.py              # Agentic Query Router
│   ├── reranker.py            # Cross-Encoder Reranking
│   ├── temporal.py            # Recency Boosting
│   └── context_compressor.py  # Chat History Compression
├── recall/
│   ├── itemcf.py              # ItemCF Recall (direction-weighted)
│   ├── usercf.py              # UserCF Recall
│   ├── swing.py               # Swing Recall (user-pair overlap)
│   ├── sasrec_recall.py       # SASRec Embedding Recall
│   ├── popularity.py          # Popularity Recall
│   ├── youtube_dnn.py         # Two-Tower Model
│   └── fusion.py              # RRF Fusion (6 channels)
├── ranking/
│   └── features.py            # 17 Ranking Features
├── data_factory/
│   └── generator.py           # SFT Data Synthesis + LLM Judge
├── services/
│   ├── chat_service.py        # RAG Chat Pipeline
│   └── recommend_service.py   # Personalized Recommendation
└── vector_db.py               # Hybrid Search + Small-to-Big
```

---

## 10. Scalability

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
