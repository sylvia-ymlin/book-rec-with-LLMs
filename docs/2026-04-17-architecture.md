# Project Architecture: Book Recommendation System with LLMs

This document describes the high-level architecture of the system as of April 2026, highlighting the dual-engine approach (RAG vs. RecSys) and the data flow across components.

## High-Level Architecture Diagram

```mermaid
flowchart TD
    User(("User / Browser")) <--> Frontend["React Frontend<br/>TypeScript + Vite"]
    
    subgraph "Application Layer (FastAPI)"
        API["API Gateway<br/>main.py"]
        
        subgraph "Search & RAG Engine"
            Orchestrator["RecommendationOrchestrator"]
            Router["Query Router<br/>Intent Classification"]
            VectorDB["VectorDB Wrapper<br/>Hybrid Search"]
            Agentic["LangGraph Workflow<br/>Agentic RAG"]
        end
        
        subgraph "Personalized RecSys Engine"
            RecService["RecommendationService"]
            Recall["Recall Fusion<br/>ItemCF, UserCF, SASRec..."]
            Ranker["Unified Ranker<br/>DIN / Stacking / LGBM"]
            Explainer["SHAP Explainer<br/>Feature Attribution"]
        end
        
        Diversity["Diversity Reranker<br/>MMR + Category Filter"]
    end
    
    subgraph "Models & LLMs"
        LLM["LLM Provider<br/>Ollama / OpenAI / Groq"]
        Embeddings["Embedding Models<br/>Sentence-Transformers"]
        CrossEnc["Cross-Encoder<br/>BGE-Reranker"]
    end
    
    subgraph "Data & Infrastructure"
        SQLite[("SQLite / FTS5<br/>Metadata & History")]
        Chroma[("ChromaDB<br/>Vector Store")]
        Redis[("Redis<br/>Cache")]
        WebFallback["Google Books API"]
        Prometheus["Monitoring<br/>Prometheus + Grafana"]
    end

    %% Flow: Frontend to API
    Frontend <--> API
    
    %% Flow: Search Path
    API --> Orchestrator
    Orchestrator --> Router
    Router --> VectorDB
    Router --> Agentic
    VectorDB --> Chroma
    VectorDB --> SQLite
    Agentic --> LLM
    Agentic --> WebFallback
    
    %% Flow: RecSys Path
    API --> RecService
    RecService --> Recall
    Recall --> SQLite
    RecService --> Ranker
    RecService --> Explainer
    Ranker --> SQLite
    
    %% Flow: Shared Logic
    Orchestrator --> Diversity
    RecService --> Diversity
    Diversity --> SQLite
    
    %% Infrastructure
    API --- Prometheus
    API --- Redis
```

## Component Breakdown

### 1. Dual-Engine Recommendation Logic
The system features two independent pipelines tailored for different user needs:
- **RAG Engine (Knowledge-based)**: Used for intent-driven natural language queries (e.g., "thriller with a twist"). It uses **Hybrid Search** (BM25 + Dense Vector) and can escalate to an **Agentic LangGraph** workflow for complex queries.
- **RecSys Engine (Personalization)**: Used for history-based discovery. It employs a **Multi-channel Recall** strategy (7 algorithms) fused into a **Unified Ranker** (supporting Deep Interest Networks or Gradient Boosting).

### 2. Intelligent Routing
The **Query Router** analyzes incoming requests to determine the best retrieval strategy:
- **Small-to-Big Search**: For general metadata queries.
- **Hybrid Search**: Fusing keyword matches (ISBN/Author) with semantic similarity.
- **Freshness Fallback**: Triggering web search if local data is insufficient.

### 3. Ranking & Diversity
- **Unified Ranking**: A hierarchy of models (DIN > Stacking > LGBM) ensures the best available performance.
- **Shapley Explainability**: Provides transparent reasons for recommendations (e.g., "Recommended because you like Thrillers (+0.15)").
- **MMR Reranking**: Ensures results are not too similar, maximizing catalog exposure and user engagement.

### 4. Data Layer (Zero-RAM Design)
The system is optimized for low memory usage:
- **SQLite + FTS5**: Handles structured metadata lookups and full-text search.
- **ChromaDB**: Manages vector embeddings for semantic search.
- **Redis Cache**: Minimizes latency for frequent queries.

### 5. Monitoring & Ops
A full observability stack is integrated:
- **Prometheus**: Tracks latencies, request counts, and model metrics.
- **Grafana**: Visualizes system health and recommendation KPIs.
- **Alertmanager**: Triggers notifications for system anomalies.
