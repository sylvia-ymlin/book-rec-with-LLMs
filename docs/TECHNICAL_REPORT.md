# Technical Report: Agentic RAG Book Recommender

**Author**: [Your Name]  
**Date**: January 2026  
**Project Type**: End-to-End ML/AI System (Retrieval-Augmented Generation)

---

## Executive Summary

This project implements a **production-grade Agentic RAG (Retrieval-Augmented Generation)** system for book discovery. Unlike simple vector search, it uses a self-routing architecture that dynamically selects the optimal retrieval strategy based on query intent, achieving:

- **100% recall** on exact-match queries (ISBNs)
- **Sub-second latency** for keyword searches
- **Deep semantic understanding** for complex natural language queries
- **Detail-level precision** via hierarchical (Small-to-Big) retrieval

The system demonstrates mastery of both **Data-Centric AI** (SFT data synthesis) and **Advanced RAG Architecture** (Hybrid Search, Reranking, Query Routing).

---

## 1. Problem Statement

**Challenge**: Traditional keyword search fails on modern book discovery scenarios:
- Users search by *feeling* ("sad sci-fi about AI") rather than *keywords*
- Users want specific *plot details* ("books with an unreliable narrator twist")
- Users expect *temporal awareness* ("latest books on quantum computing")

**Solution**: An intelligent RAG system that:
1. Understands user intent (Agentic Router)
2. Fuses multiple retrieval strategies (Hybrid Search)
3. Ranks results by semantic relevance (Cross-Encoder Reranking)
4. Finds hidden gems in review text (Small-to-Big Retrieval)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     AGENTIC QUERY ROUTER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │    ISBN     │  │   Keyword   │  │   Complex   │  │   Detail    │    │
│  │   (Exact)   │  │   (Fast)    │  │   (Deep)    │  │ (Small2Big) │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼───────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
    ┌──────────┐    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ BM25 Only│    │Hybrid (RRF)  │  │Hybrid + Rank │  │Chunk → Parent│
    │ α=1.0    │    │BM25 + Dense  │  │+ Cross-Enc   │  │788K Sentences│
    └────┬─────┘    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                 │                 │
         └─────────────────┴────────┬────────┴─────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    OPTIONAL POST-PROCESSING                             │
│  ┌──────────────────┐    ┌──────────────────┐                          │
│  │ Temporal Dynamics│    │Context Compression│                          │
│  │ (Recency Boost)  │    │  (Chat History)   │                          │
│  └──────────────────┘    └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LLM GENERATION                                  │
│              (Streaming Response via SSE)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technical Innovations

### 3.1 Agentic Query Router (`src/core/router.py`)

**Motivation**: A single retrieval strategy cannot optimize for all query types.

**Implementation**: Rule-based intent classifier using RegEx and keyword detection:
| Query Type | Detection Logic | Strategy | Latency |
|------------|-----------------|----------|---------|
| ISBN | `\d{10,13}` pattern | BM25 Only (α=1.0) | <100ms |
| Keyword | `len(words) <= 2` | Hybrid (No Rerank) | ~300ms |
| Complex | Default | Hybrid + Cross-Encoder | ~800ms |
| Detail | Keywords: "twist", "ending", "cried" | Small-to-Big | ~500ms |

**Trade-off Decision**: Chose rule-based over LLM-based routing to avoid 500ms+ latency per routing decision.

### 3.2 Hybrid Search with RRF (`src/vector_db.py`)

**Motivation**: Dense vectors fail on exact terms (ISBNs, proper nouns); BM25 fails on semantic queries.

**Implementation**: Reciprocal Rank Fusion combining BM25 (sparse) and MiniLM (dense):
```python
RRF_Score = Σ 1/(k + rank_dense) + 1/(k + rank_sparse)  # k=60
```

**Result**: 100% recall on ISBNs (previously 0% with pure vector search).

### 3.3 Cross-Encoder Reranking (`src/core/reranker.py`)

**Motivation**: Bi-encoders are fast but approximate; Cross-Encoders are slow but precise.

**Implementation**: Two-stage retrieval:
1. Stage 1: Retrieve top-50 candidates via RRF (~100ms)
2. Stage 2: Rerank with `ms-marco-MiniLM-L-6-v2` (~400ms)

**Trade-off Decision**: Only rerank top-50 (not all 200K) to balance precision vs latency.

### 3.4 Small-to-Big Retrieval (`src/vector_db.py::small_to_big_search`)

**Motivation**: Book descriptions are coarse; review sentences contain fine-grained details.

**Implementation** (SOTA: LlamaIndex Parent-Child, RAPTOR):
1. **Chunking**: 788,174 review sentences indexed at sentence-level
2. **Matching**: Query matches specific sentence ("I cried at the ending")
3. **Expansion**: Map sentence → parent ISBN → full book context

**Result**: Can answer queries like "books with unreliable narrator twist" that are invisible to description-level search.

### 3.5 SFT Data Factory (`src/data_factory/generator.py`)

**Motivation**: Default LLM tone is corporate; we want "Literary Critic" personality.

**Implementation** (SOTA: Self-Instruct, Alpaca):
1. **Seed Sampling**: Extract 1000 high-emotion reviews (rating=5, length>200)
2. **Instruction Evolution**: GPT generates user questions that would prompt each review
3. **Response Transform**: Rewrite reviews as AI assistant style
4. **LLM-as-a-Judge**: Filter for Empathy/Specificity/Critique Depth >= 8/10

**Output**: Production-ready SFT dataset for style alignment.

---

## 4. Performance Metrics

| Metric | Baseline (Vector Only) | Advanced (This System) |
|--------|------------------------|------------------------|
| ISBN Recall | 0% | **100%** |
| Keyword Precision | Low | **High** (BM25 boost) |
| Detail Query Recall | 0% | **High** (Small-to-Big) |
| Avg Latency | 100ms | 300-800ms (acceptable) |
| Chat Context Limit | ~10 turns | **Unlimited** (compression) |

---

## 5. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Vector DB** | ChromaDB | Embedded, zero-latency vector storage |
| **Sparse Index** | BM25Okapi (rank_bm25) | Keyword/exact match retrieval |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim sentence embeddings |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Cross-encoder precision ranking |
| **LLM** | OpenAI / Ollama (llama3) | Generation with BYOK support |
| **Backend** | FastAPI + SSE | Streaming API |
| **Frontend** | React 18 + Vite | Modern SPA |

---

## 6. Key Design Decisions

| Decision | Chosen Option | Rejected Alternative | Rationale |
|----------|---------------|---------------------|-----------|
| Vector DB | ChromaDB (embedded) | Pinecone (cloud) | Zero network latency; 200K docs fits in RAM |
| Routing | Rule-based RegEx | LLM-based routing | 2ms vs 500ms latency; deterministic behavior |
| Reranking | Cross-Encoder | LLM reranking | 400ms vs 2s latency; proven accuracy |
| Chunking | Sentence-level (Small-to-Big) | Fixed 512 tokens | Semantic integrity; detail-level matching |
| SFT Data | Self-Instruct | Manual annotation | Scalable; leverages existing reviews |

---

## 7. Interview Talking Points

**Q: What makes this project technically interesting?**
> "I implemented an Agentic RAG system with self-routing capability. Instead of one-size-fits-all vector search, the system classifies query intent and dynamically selects from 4 strategies—each optimized for different query types. This achieved 100% recall on exact-match queries that previously failed."

**Q: What was the hardest engineering challenge?**
> "The Small-to-Big retrieval. I indexed 788K review sentences separately, but the challenge was mapping matched sentences back to their parent books efficiently. I solved it by embedding parent ISBN in chunk metadata and using BM25 for O(1) lookup."

**Q: How would you improve this further?**
> "Three directions: (1) Fine-tune embeddings on book domain for better semantic alignment, (2) Implement HyDE (generate hypothetical documents before searching), (3) Add RAGAS evaluation pipeline for systematic quality measurement."

---

## 8. File Structure

```
src/
├── core/
│   ├── router.py           # Agentic Query Router
│   ├── reranker.py         # Cross-Encoder Reranking
│   ├── temporal.py         # Recency Boosting
│   └── context_compressor.py # Chat History Compression
├── data_factory/
│   └── generator.py        # SFT Data Synthesis + LLM Judge
├── vector_db.py            # Hybrid Search + Small-to-Big
├── recommender.py          # Main recommendation logic
└── services/chat_service.py # RAG Chat Pipeline

docs/
├── TECHNICAL_REPORT.md     # This document
├── technical_deep_dive_sota.md # SOTA references
├── rag_architecture.md     # System diagrams
└── interview_deep_dive.md  # Interview prep

experiments/
├── baseline_report.md      # Dense-only baseline
├── hybrid_report.md        # Hybrid search results
├── rerank_report.md        # Cross-encoder results
├── router_report.md        # Agentic router results
└── temporal_report.md      # Time decay results
```

---

## 9. Conclusion

This project demonstrates end-to-end ML engineering skills across:
- **Data Engineering**: ETL pipelines, SFT data synthesis, quality filtering
- **ML Systems**: Hybrid retrieval, cross-encoder reranking, hierarchical indexing
- **Production Engineering**: Streaming APIs, caching, context management
- **Architecture Design**: Trade-off analysis, performance optimization

The system is **production-ready** and serves as a strong portfolio piece for MLE/AI Engineer roles.

---

## References

1. Self-Instruct (Wang et al., 2022) - Instruction data synthesis
2. RAPTOR (Sarthi et al., 2024) - Hierarchical tree-based indexing
3. HyDE (Gao et al., 2022) - Hypothetical document embeddings
4. LlamaIndex - Parent-child retrieval patterns
5. ms-marco-MiniLM - Cross-encoder reranking
