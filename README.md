# Book Recommendation with LLMs

A book recommendation system that combines a RAG pipeline (for query-driven search) and a collaborative filtering pipeline (for history-based personalization), backed by a FastAPI server and a React UI.

## What it does

Given a natural language query (e.g. "a thriller with an unreliable narrator"), the system retrieves relevant books using hybrid search (BM25 keyword + dense vector) and reranks results with a cross-encoder. For users with a reading history, it also provides personalized recommendations via multi-channel collaborative filtering ranked by LGBMRanker. A "Chat with Book" feature streams LLM answers grounded in book metadata.

## Problem it addresses

**Pure vector search fails on exact identifiers.** Searching by ISBN or author name returns wrong results because sentence embeddings don't preserve lexical identity. This system routes queries to sparse (BM25) or hybrid retrieval depending on query type, then applies cross-encoder reranking to correct semantic drift (e.g. "Harry Potter" → wrong-genre book).

**A single collaborative filtering algorithm misses many relevant books.** Seven recall channels (ItemCF, UserCF, Swing, Item2Vec, SASRec, YouTube-DNN, popularity) broaden candidate coverage, and LGBMRanker learns which signals matter per user.

## Approach

Two independent recommendation paths:

**RAG path** — for query-driven lookup:
1. Query router classifies intent (detail-focused, freshness-focused, or natural language)
2. Hybrid search: BM25 keyword search (FTS5) + dense embedding search (ChromaDB), fused with Reciprocal Rank Fusion
3. Cross-encoder reranking for precision
4. Optional LangGraph agentic workflow (router → retrieve → evaluate → web fallback) for complex queries

**RecSys path** — for history-based personalization:
1. Seven recall channels generate ~1000 candidates from user history
2. LGBMRanker (LambdaRank) scores candidates using features from all channels
3. Optional MMR diversity reranking to reduce result homogeneity

## Project structure

```
src/
  app/
    main.py          App factory: startup, middleware, Prometheus, frontend serving
    state.py         Shared singleton state (recommender, rec_service)
    api/             Route modules
      recommend.py   /recommend, /api/recommend/similar, /api/recommend/personal
      bookshelf.py   /favorites/*, /categories, /api/onboarding/books
      admin.py       /books/add, /benchmark, /marketing/highlights
      chat.py        /chat/completions (streaming)
      ops.py         /ops/kpi, /ops/cache/*
  core/              Business logic (orchestrator, intent classifier, diversity reranker)
  rag/               Retrieval: hybrid search, reranker, context compressor, router
  recsys/
    recall/          Recall channels (ItemCF, UserCF, Swing, Item2Vec, SASRec)
    ranking/         LGBMRanker, feature engineering
  services/          Service layer (chat, recommendation)
  support/
    agentic/         LangGraph workflow (router → retrieve → LLM evaluate → web fallback)
    marketing/       Personalized marketing copy generation
  data/
    stores/          SQLite metadata store, profile store, online books staging
    repository.py    Data access layer (metadata + user history from recall_models.db)
  infra/             Config, Redis cache, logging utilities
  model/             LGBMRanker model
web/                 React frontend (TypeScript + Vite)
data/
  chroma_db/         ChromaDB vector index
  books.db           SQLite book metadata + FTS5 full-text index
config/
  router.json        Router keyword configuration
deploy/              Prometheus, Alertmanager, Grafana configuration
```

## How to run

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the frontend)
- [Ollama](https://ollama.ai) for local LLM, or an OpenAI/Groq API key

### Local setup

```bash
git clone https://github.com/sylvia-ymlin/book-rec-with-LLMs.git
cd book-rec-with-LLMs

# Python environment
conda env create -f environment.yml
conda activate book-rec

# Copy and configure environment variables
cp .env.example .env   # add API keys if using OpenAI or Groq

# Initialize databases (first run only)
python data/scripts/init_db.py        # builds ChromaDB vector index
python scripts/init_sqlite_db.py      # builds SQLite metadata + FTS5 index

# Start the API server
make run                              # http://localhost:6006

# Start the frontend (separate terminal)
cd web && npm install && npm run dev  # http://localhost:5173
```

### Docker

```bash
docker-compose up
```

Starts: API (port 8000), Redis, Prometheus (port 9090), Alertmanager, Grafana (port 3000).

## Example output

`POST /recommend` — `{"query": "a thriller with an unreliable narrator", "category": "Fiction"}`:

```json
{
  "recommendations": [
    {
      "isbn": "9780385333481",
      "title": "Gone Girl",
      "authors": "Gillian Flynn",
      "description": "A psychological thriller about a missing wife...",
      "thumbnail": "/content/9780385333481.jpg",
      "caption": "Twisty psychological thriller with unreliable perspectives",
      "tags": ["thriller", "suspense", "mystery"],
      "average_rating": 4.0,
      "explanations": []
    }
  ]
}
```

`GET /api/recommend/personal?user_id=alice&top_k=5` returns personalized books ranked from Alice's reading history.

## Evaluation results

| Experiment | Before | After | Conclusion |
|:-----------|:-------|:------|:-----------|
| RAG: Exact match | Pure vector search, ISBN → 0% recall | Hybrid (BM25 + Dense) + Router → 100% | Vector-only fails on exact entities; BM25 + routing fixes it |
| RAG: Keyword intent | "Harry Potter" → Philosophy book | Reranked → Sorcerer's Stone | Cross-encoder corrects semantic drift |
| RecSys: Personalization | Baseline HR@10 = 0.138 | Item2Vec + LGBMRanker + Stacking → HR@10 = **0.4545** | 7-channel recall + LambdaRank beats single-model baseline |

*Evaluation: Leave-Last-Out protocol, n=2000, title-relaxed matching. HR@10 = 0.4545, MRR@5 = 0.2893.*

## Limitations

- **Static book corpus**: The system recommends only from the local SQLite + ChromaDB index. Adding new books requires re-indexing. The web fallback (Google Books API) covers some gaps but is rate-limited.
- **RecSys needs history**: The personalized path falls back to popularity-based recommendations for new users with no reading history.
- **Local LLM required for offline use**: Chat and LLM-evaluate features use Ollama by default; cloud providers (OpenAI, Groq) need API keys.
- **English-primary**: Retrieval and reranking are optimized for English queries; other languages may give degraded results.
- **No real-time freshness**: Publication dates come from the static dataset; there is no live feed for new releases.

## License

MIT
