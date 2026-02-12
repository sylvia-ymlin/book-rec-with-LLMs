---
license: mit
title: Semantic-Based Book Recommendation Framework
sdk: docker
app_port: 8000
---

# Intelligent Book Recommendation System

*Frozen at v2.6.0 — maintenance mode for portfolio use.*

## Problem

Readers often can't articulate what they want. Can one system both understand their vague descriptions and give personalized recommendations based on their reading history?

## Method

Two parallel threads: **RAG** (Agentic Router → Hybrid Search → Reranking) for understanding vague queries; **RecSys** (7-channel recall → LGBMRanker → Stacking) for personalized recommendations from reading history.

## Key Experiments

| Experiment | Before | After | Conclusion |
|:---|:---|:---|:---|
| **RAG: Exact match** | Pure vector search, ISBN → 0% recall | Hybrid (BM25 + Dense) + Router → 100% | Vector-only fails on exact entities; BM25 + routing fixes it |
| **RAG: Keyword intent** | "Harry Potter" → Philosophy book | Reranked → Sorcerer's Stone | Cross-encoder corrects semantic drift |
| **RecSys: Personalization** | Baseline 0.138 HR@10 | Item2Vec + LGBMRanker + Stacking → **0.4545** HR@10 | 7-channel recall + LambdaRank + ensemble beats single model |

*Evaluation: Leave-Last-Out, n=2000, title-relaxed. HR@10 = 0.4545, MRR@5 = 0.2893.*

## Architecture

```
         Query                    No Query
            │                         │
            ▼                         ▼
   ┌─────────────┐            ┌─────────────┐
   │  RAG Path  │            │ RecSys Path │
   │ Router →   │            │ 7-Channel   │
   │ Hybrid →   │            │ Recall →    │
   │ Rerank     │            │ LGBMRanker  │
   └─────────────┘            └─────────────┘
            │                         │
            └──────────┬──────────────┘
                       ▼
                 Top-K Results
```

## Quick Start

```bash
git clone https://github.com/sylvia-ymlin/book-rec-with-LLMs.git
cd book-rec-with-LLMs
conda env create -f environment.yml && conda activate book-rec

# First run (or use make data-pipeline for full build)
python src/init_db.py              # Chroma vector DB
python scripts/init_sqlite_db.py    # SQLite metadata (local build)

make run                       # API http://localhost:6006
cd web && npm install && npm run dev   # UI http://localhost:5173
```

**LLM**: Default Ollama (`ollama pull llama3`). OpenAI API key in UI Settings for production.

## Documentation

| Doc | Purpose |
|:---|:---|
| [Technical Report](docs/TECHNICAL_REPORT.md) | Architecture, design decisions |
| [Development Guide](docs/DEVELOPMENT.md) | 添加召回通道、调整路由规则 |
| [Contributing](CONTRIBUTING.md) | 贡献者指南 |
| [Experiment Archive](docs/experiments/experiment_archive.md) | Full experiment log (V1.0 → v2.6.0) |
| [Interview Guide](docs/interview_guide.md) | Q&A, STAR cases |
| [Build Guide](docs/build_guide.md) | Deployment instructions |

## License

MIT
