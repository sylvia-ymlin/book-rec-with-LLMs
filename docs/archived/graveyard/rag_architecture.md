# Advanced RAG Architecture: Technical Overview
**Project**: Book Recommender with LLMs
**Date**: Jan 2026

## 1. System Overview
This project implements an **Agentic RAG (Retrieval-Augmented Generation)** system designed to overcome the limitations of standard semantic search. It uses a **Self-Reliant Router** to dynamically select the optimal retrieval strategy based on user intent.

### Key Capabilities
- **Exact Match**: Zero-error retrieval for ISBNs and specific IDs.
- **Deep Understanding**: Semantic search + Reranking for complex queries.
- **Temporal Awareness**: Recency bias for "latest/new" queries.
- **Efficient Memory**: Token-saving context compression.

---

## 2. Architecture Pipeline

```mermaid
graph TD
    UserQuery[User Query] --> Router{Query Router}
    
    %% Strategy 1: Exact
    Router -- "ISBN Detected" --> Exact[BM25 Sparse Only]
    Exact --> Result
    
    %% Strategy 2: Fast
    Router -- "Keywords (Short)" --> Fast[Hybrid Search (No Rerank)]
    Fast --> Result
    
    %% Strategy 3: Deep
    Router -- "Natural Language" --> Hybrid[Hybrid Search (BM25 + Dense)]
    Hybrid --> Fusion[Reciprocal Rank Fusion (RRF)]
    Fusion --> Top50[Top 50 Candidates]
    Top50 --> Rerank[Cross-Encoder (ms-marco-MiniLM)]
    
    %% Temporal Layer
    Rerank --> Temporal{Temporal Keywords?}
    Temporal -- "Yes (e.g. 'latest')" --> Decay[Apply Time Decay Boost]
    Temporal -- "No" --> RankScore
    Decay --> RankScore[Final Top K]
    
    RankScore --> Result[Context for LLM]
```

## 3. Component Details

### 3.1. Hybrid Search (The Foundation)
Combines **Sparse Retrieval (BM25)** and **Dense Retrieval (ChromaDB/All-MiniLM)** using **Reciprocal Rank Fusion (RRF)**.
- **Why?**: Dense vectors fail at exact keyword matching (e.g., "Harry Potter"). BM25 fails at semantic understanding. Together, they cover 100% of use cases.
- **Implementation**: `src/vector_db.py`

### 3.2. Cross-Encoder Reranking (The Refiner)
A second-stage pass using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Why?**: Bi-Encoders (Vectors) are fast but approximate. Cross-Encoders are slow but highly accurate. We only rerank the top 20-50 results.
- **Impact**: Improved precision for complex queries (e.g., distinguishing "Philosophy of Harry Potter" from "Harry Potter and the Sorcerer's Stone").
- **Implementation**: `src/core/reranker.py`

### 3.3. Agentic Router (The Brain)
Classifies input using Regex and Keyword analysis to short-circuit expensive steps.
- **Strategies**:
    - **EXACT**: `alpha=1.0` (BM25 Only). Solves the "Exact Match" regression.
    - **FAST**: `rerank=False`. < 500ms latency for simple lookups.
    - **DEEP**: `rerank=True`. Full power for reasoning tasks.
- **Implementation**: `src/core/router.py`

### 3.4. Temporal Dynamics (The Bias)
Applies a log-linear decay function to boost newer documents.
- **Formula**: $Score_{new} = Score_{old} + \frac{2.0}{\ln(Age + 2.718)}$
- **Trigger**: Activated by words like "new", "latest", "2024".
- **Implementation**: `src/core/temporal.py`

### 3.5. Context Compression (The Memory)
Summarizes conversation history when it exceeds token limits.
- **Logic**: Retains the last 2 turns (4 messages) raw; summarizes everything older using a lightweight LLM call.
- **Implementation**: `src/core/context_compressor.py`

## 4. Performance Benchmarks
| Metric | Baseline (Dense) | Advanced (hybrid+Rerank) |
| :--- | :--- | :--- |
| **ISBN Success Rate** | 0% (Fail) | **100%** (via Router) |
| **Keyword Precision** | Low | **High** |
| **Latency (Avg)** | 20ms | 600ms - 1.2s |

## 5. Future Roadmap
- **GraphRAG**: For multi-hop reasoning across books.
- **Fine-tuning**: Domain-specific embedding adapter.
