---
license: mit
title: Semantic-Based Book Recommendation Framework
sdk: docker
app_port: 8000
---

# Intelligent Book Recommendation System

> A production-grade **Agentic RAG + RecSys** platform combining semantic search, personalized recommendations, and conversational AI.

## Highlights

| Component | Technology | Achievement |
|:---|:---|:---|
| **Semantic Search** | ChromaDB + MiniLM-L6 | Sub-300ms retrieval on 200K+ books |
| **Agentic Router** | Rule-based intent classification | 4 dynamic strategies (BM25, Hybrid, Rerank, Small-to-Big) |
| **Personalized Rec** | SASRec + XGBoost | MRR@5: 0.21, HR@10: 0.44 |
| **Conversational AI** | RAG + OpenAI/Ollama | Real-time streaming (Default: Local Ollama) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│   Search UI │ My Bookshelf │ Chat │ Recommendations             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST + SSE
┌──────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Backend                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐   │
│  │ Query Router│→ │ RAG Pipeline │→ │ Personalized RecSys   │   │
│  └─────────────┘  └──────────────┘  └───────────────────────┘   │
│         │                │                    │                  │
│    Intent Class    Hybrid Search      Multi-Channel Recall      │
│    (ISBN/Keyword    + Cross-Encoder   (ItemCF + UserCF +        │
│     /Complex)       Reranking         SASRec + Popularity)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐      ┌───────────┐      ┌──────────────┐
   │ChromaDB │      │ XGBoost   │      │ LLM Provider │
   │(Vectors)│      │ (Ranking) │      │ (Chat/Recs)  │
   └─────────┘      └───────────┘      └──────────────┘
```

---

## Key Features

### 1. Agentic RAG with Dynamic Routing
- **Query Intent Classification**: Automatically routes queries to optimal retrieval strategy
  - ISBN → Pure BM25 (100% precision)
  - Keywords → Hybrid Search (BM25 + Dense)
  - Complex queries → Cross-Encoder Reranking
  - Detail queries → Small-to-Big Retrieval (788K indexed sentences)

### 2. Personalized Recommendation Engine
- **Multi-Channel Recall**: ItemCF, UserCF, Popularity
- **SASRec Sequential Model**: 64-dim Transformer embeddings (30 epochs)
- **XGBoost Ranker**: Feature-based ranking with learned weights
- **Evaluation Results**: MRR@5 = 0.2089, Hit Rate@10 = 0.4400

### 3. My Bookshelf (User Library)
- **Rating System**: 5-star rating with persistence
- **Reading Status**: Want to Read / Reading / Finished
- **Statistics Dashboard**: Visual progress tracking

### 4. Conversational Shopping Assistant
- **RAG-Grounded Responses**: Context from ChromaDB reduces hallucinations
- **Streaming Output**: Real-time token streaming via SSE
- **Flexible LLM**: Defaults to local **Ollama** (free/privacy), supports **OpenAI API** for production

---

## Quick Start

### Prerequisites
- Python 3.10+ with Conda
- Node.js 18+

### Installation

```bash
# Clone and setup environment
git clone https://github.com/sylvia-ymlin/book-rec-with-LLMs.git
cd book-rec-with-LLMs
conda env create -f environment.yml
conda activate book-rec

# Initialize vector database (first run)
python src/init_db.py

# Start API server
make run  # http://localhost:6006

# Start frontend (new terminal)
cd web && npm install && npm run dev  # http://localhost:5173
```

### LLM Configuration

| Provider | Setup | Use Case |
|:---|:---|:---|
| **Ollama** | `ollama pull llama3` | Free, local dev |
| **OpenAI** | Set API key in UI Settings | Production |

---

## API Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/recommend` | POST | Semantic search with emotion/category filters |
| `/api/recommend/personal` | GET | Personalized recommendations (RecSys) |
| `/favorites/add` | POST | Add book to collection |
| `/favorites/update` | PUT | Update rating/reading status |
| `/user/{id}/stats` | GET | Reading statistics |
| `/chat/completions` | POST | RAG-powered chat (streaming) |
| `/health` | GET | Service health check |

---

## Project Documentation

For a detailed analysis of the system architecture, experimental results, and engineering decisions, please refer to the following academic-style reports:

- [Interview Playbook](docs/interview_playbook.md): Core problem analysis, S.T.A.R. cases, and engineering trade-offs.
- [Technical Report](docs/technical_report.md): Deep dive into system architecture, RAG strategies, and RecSys pipeline.
- [Experiment Report](docs/experiment_report.md): Performance benchmarks, model evaluation (SASRec/XGBoost), and latency tests.

---

## Project Structure

```
src/
├── main.py              # FastAPI application
├── recommender.py       # RAG search orchestration
├── vector_db.py         # ChromaDB wrapper
├── core/
│   ├── router.py        # Agentic query routing
│   └── reranker.py      # Cross-encoder reranking
├── recall/              # RecSys recall channels (ItemCF, SASRec, etc.)
├── ranking/             # XGBoost ranking features
├── services/            # Recommendation service
└── user/                # User profile storage

web/
├── src/App.jsx          # React UI
└── src/api.js           # API client

scripts/
├── model/
│   ├── train_sasrec.py      # SASRec model training
│   ├── train_ranker.py      # XGBoost ranker training
│   └── evaluate.py          # Evaluation metrics
├── deploy/                  # Server deployment scripts
└── data/                    # Data processing pipelines
```

---

## Performance

### Recommendation Metrics
| Metric | Value | Notes |
|:---|:---|:---|
| **Hit Rate@10** | 0.4400 | Target book in top-10 |
| **MRR@5** | 0.2089 | Mean Reciprocal Rank (strict) |
| Dataset Size | ~168K Users | ~152K Books with ratings |

### Latency Benchmarks
| Operation | P50 Latency |
|:---|:---|
| **Exact Search** | ~19ms |
| **Hybrid Search** | ~230ms |
| **Reranked Search** | ~710ms |

---

## References

1. Kang, W., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. ICDM.
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
4. Gao, L., et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*.

---

## License

MIT

