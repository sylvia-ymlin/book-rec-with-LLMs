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
| **Personalized Rec** | 6-channel recall + LGBMRanker | HR@10: 0.2205, MRR@5: 0.1584 |
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
│    (ISBN/Keyword    + Cross-Encoder   (ItemCF + UserCF + Swing  │
│     /Complex)       Reranking         + SASRec + Popularity)    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐      ┌───────────┐      ┌──────────────┐
   │ChromaDB │      │LGBMRanker │      │ LLM Provider │
   │(Vectors)│      │(LambdaRank│      │ (Chat/Recs)  │
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
- **6-Channel Recall**: ItemCF (direction-weighted), UserCF, Swing, SASRec, YoutubeDNN, Popularity
- **RRF Fusion**: Reciprocal Rank Fusion merges candidates across all recall channels
- **SASRec Sequential Model**: 64-dim Transformer embeddings (30 epochs), used as both recall source and ranking feature
- **LGBMRanker (LambdaRank)**: Directly optimizes NDCG with 17 engineered features and hard negative sampling
- **Evaluation**: HR@10 = 0.2205, MRR@5 = 0.1584 (n=2000, Leave-Last-Out)

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

## Project Structure

```
src/
├── main.py              # FastAPI application
├── recommender.py       # RAG search orchestration
├── vector_db.py         # ChromaDB wrapper
├── core/
│   ├── router.py        # Agentic query routing
│   └── reranker.py      # Cross-encoder reranking
├── recall/
│   ├── itemcf.py        # ItemCF with direction weight
│   ├── usercf.py        # UserCF (Jaccard + activity penalty)
│   ├── swing.py         # Swing (user-pair overlap weighting)
│   ├── sasrec_recall.py # SASRec embedding dot-product recall
│   ├── youtube_dnn.py   # YoutubeDNN two-tower recall
│   ├── popularity.py    # Popularity with time decay
│   └── fusion.py        # RRF fusion of all channels
├── ranking/
│   └── features.py      # 17 ranking features
├── services/
│   └── recommend_service.py  # Recall → Rank → Dedup pipeline
└── user/                # User profile storage

web/
├── src/App.jsx          # React UI
└── src/api.js           # API client

scripts/
├── model/
│   ├── train_sasrec.py          # SASRec sequential model training
│   ├── build_recall_models.py   # ItemCF, UserCF, Swing, Popularity
│   ├── train_ranker.py          # LGBMRanker with hard negative sampling
│   └── evaluate.py              # HR@10, MRR@5 evaluation
├── deploy/                      # Server deployment scripts
└── data/                        # Data processing pipelines
```

---

## Performance

### Recommendation Metrics (V2.5)

| Metric | V2.0 | V2.5 | Method |
|:---|:---|:---|:---|
| **Hit Rate@10** | 0.1380 | **0.2205** (+59.8%) | Leave-Last-Out, n=2000 |
| **MRR@5** | 0.1295 | **0.1584** (+22.3%) | Title-relaxed matching |

V2.5 key changes: +ItemCF direction weight, +Swing recall, +SASRec recall channel, XGBoost→LGBMRanker (LambdaRank), random→hard negative sampling.

| Dataset | Size |
|:---|:---|
| Training Set | 1,079,966 interactions |
| Active Users | 167,968 |
| Books | 221,998 |

### Latency Benchmarks
| Operation | P50 Latency |
|:---|:---|
| **Exact Search** | ~19ms |
| **Hybrid Search** | ~230ms |
| **Reranked Search** | ~710ms |
| **Personal Rec (warm)** | ~19ms |

---

## Project Documentation

| Document | Description |
|:---|:---|
| [Experiment Archive](docs/experiments/experiment_archive.md) | All experimental results from V1.0 to V2.5 |
| [Performance Debugging Report](docs/performance_debugging_report.md) | Root cause analysis of evaluation issues |
| [Roadmap](docs/roadmap.md) | Technical evolution plan (V2.0 → V3.0) |
| [Technical Report](docs/technical_report.md) | System architecture deep dive |
| [Build Guide](docs/build_guide.md) | Build and deployment instructions |

## References

1. Kang, W., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. ICDM.
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
3. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
4. Gao, L., et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*.
5. Yang, J., et al. (2020). *Large-scale Product Graph Construction for Recommendation in E-commerce* (Swing algorithm).

---

## License

MIT

