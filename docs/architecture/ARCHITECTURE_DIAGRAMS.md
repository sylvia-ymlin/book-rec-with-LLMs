# 架构图可视化

本文档补充 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) 中的 ASCII 架构图，提供可渲染的流程图、时序图和 ER 图。使用 [Mermaid](https://mermaid.js.org/) 语法，可在 GitHub、GitLab、VS Code 等环境中直接渲染。

---

## 1. 流程图 (Flowcharts)

### 1.1 RAG 检索流水线

```mermaid
flowchart TD
    A[User Query] --> Router{Agentic Query Router}

    Router -->|ISBN 模式| S1[BM25 Only]
    Router -->|Keyword ≤2词| S2[Hybrid RRF]
    Router -->|Detail 关键词| S3[Small-to-Big]
    Router -->|默认 Complex| S4[Hybrid + Rerank]

    S1 --> Merge
    S2 --> Merge
    S3 --> Merge
    S4 --> Merge

    Merge[合并候选] --> PP[Temporal Dynamics]
    PP --> CC[Context Compression]
    CC --> LLM[LLM Generation]
    LLM --> O[Streaming Response]
```

### 1.2 个性化推荐流水线

```mermaid
flowchart TD
    subgraph Input
        A[User Request<br/>无 Query]
    end

    subgraph Recall["7-Channel Recall"]
        C1[ItemCF]
        C2[UserCF]
        C3[Swing]
        C4[SASRec]
        C5[Item2Vec]
        C6[YoutubeDNN]
        C7[Popularity]
    end

    A --> C1 & C2 & C3 & C4 & C5 & C6 & C7

    subgraph Fusion["RRF Fusion"]
        RRF[Reciprocal Rank Fusion]
    end

    C1 & C2 & C3 & C4 & C5 & C6 & C7 --> RRF

    subgraph Features["特征工程"]
        FE[User Stats, Item Stats<br/>SASRec Score, CF Scores<br/>Author/Category Affinity]
    end

    RRF --> FE

    subgraph Ranking["排序"]
        LGB[LGBMRanker<br/>LambdaRank]
        XGB[XGBClassifier]
        LR[LogisticRegression<br/>Meta-Learner]
        LGB --> Stack{Stacking?}
        XGB --> Stack
        Stack -->|是| LR
        Stack -->|否| LGB
    end

    FE --> LGB & XGB

    subgraph Diversity["多样性重排"]
        DR[MMR + Popularity Penalty<br/>Max 3 per Category]
    end

    Stack --> DR
    LGB --> DR

    DR --> O[Top-K Results]
```

### 1.3 双路径决策流程

```mermaid
flowchart LR
    subgraph Entry["用户入口"]
        Q[有 Query?]
    end

    Q -->|是| RAG[RAG Path]
    Q -->|否| Rec[RecSys Path]

    subgraph RAGPath["RAG Path"]
        R1[Router]
        R2[Hybrid Search]
        R3[Rerank]
        R1 --> R2 --> R3
    end

    subgraph RecPath["RecSys Path"]
        R4[7-Channel Recall]
        R5[LGBMRanker]
        R6[Diversity Rerank]
        R4 --> R5 --> R6
    end

    RAG --> RAGPath
    Rec --> RecPath

    RAGPath --> Merge[Top-K Results]
    RecPath --> Merge
```

---

## 2. 时序图 (Sequence Diagrams)

### 2.1 RAG 推荐接口 (`POST /recommend`)

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Cache
    participant Orchestrator
    participant Router
    participant VectorDB
    participant Reranker
    participant Enricher

    Client->>API: POST /recommend {query, category}
    API->>Orchestrator: get_recommendations()

    Orchestrator->>Cache: get(cache_key)
    alt Cache Hit
        Cache-->>Orchestrator: cached results
        Orchestrator-->>API: results
    else Cache Miss
        Orchestrator->>Router: route(query)
        Router-->>Orchestrator: RouterDecision

        Orchestrator->>VectorDB: hybrid_search / small_to_big
        VectorDB-->>Orchestrator: ISBNs + scores

        alt Rerank Enabled
            Orchestrator->>Reranker: rerank(query, candidates)
            Reranker-->>Orchestrator: reranked list
        end

        Orchestrator->>Enricher: enrich_and_format(isbns)
        Enricher->>VectorDB: get_book_details
        Enricher-->>Orchestrator: BookResponseDict[]

        Orchestrator->>Cache: set(cache_key, results)
        Orchestrator-->>API: results
    end

    API-->>Client: {recommendations: [...]}
```

### 2.2 聊天接口 (`POST /chat/completions`)

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant ChatService
    participant DataRepo
    participant LLM
    participant Compressor

    Client->>API: POST /chat/completions {isbn, query}
    API->>ChatService: chat_stream(isbn, query)

    ChatService->>DataRepo: get_book_metadata(isbn)
    DataRepo-->>ChatService: book context

    ChatService->>ChatService: _format_book_info(book)

    loop 流式生成
        ChatService->>Compressor: compress_history(if needed)
        Compressor-->>ChatService: context window
        ChatService->>LLM: invoke(messages)
        LLM-->>ChatService: chunk
        ChatService-->>Client: SSE chunk
    end
```

### 2.3 个性化推荐接口 (`GET /api/recommend/personal`)

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant RecService
    participant Fusion
    participant FeatureEng
    participant Ranker
    participant DiversityReranker
    participant MetadataStore

    Client->>API: GET /api/recommend/personal?user_id=...
    API->>RecService: get_recommendations(user_id, top_k)

    RecService->>Fusion: recall(user_id, real_time_seq)
    Note over Fusion: 7-Channel RRF
    Fusion-->>RecService: candidate ISBNs

    RecService->>FeatureEng: build_features(user_id, candidates)
    FeatureEng-->>RecService: feature matrix

    RecService->>Ranker: predict(features)
    Ranker-->>RecService: ranked scores

    RecService->>DiversityReranker: rerank(candidates)
    DiversityReranker->>MetadataStore: get_book_metadata
    DiversityReranker-->>RecService: diversified list

    RecService-->>API: top_k recommendations
    API-->>Client: {recommendations: [...]}
```

---

## 3. ER 图 (Entity Relationship Diagram)

```mermaid
erDiagram
    books ||--o{ books_fts : "FTS5索引"
    books {
        string isbn13 PK
        string isbn10
        string title
        string authors
        text description
        string simple_categories
        string thumbnail
        float average_rating
        float joy sadness fear anger surprise
        text tags
        text review_highlights
    }

    online_books {
        string isbn13 PK
        string isbn10
        string title
        string authors
        text description
        string simple_categories
        string thumbnail
        string source
    }

    user_profiles {
        string user_id PK
        json favorites
        json cached_highlights
    }

    item_similarity {
        string item1
        string item2
        float score
    }

    user_history {
        string user_id
        string isbn
    }

    books ||--o{ item_similarity : "item1/item2"
    books ||--o{ user_history : "isbn"
    user_profiles }o--o{ books : "收藏 favorites"
```

### 数据存储说明

| 存储 | 路径 | 用途 |
|------|------|------|
| `books` | `data/books.db` | 主元数据，只读 |
| `books_fts` | `data/books.db` | FTS5 全文搜索 |
| `online_books` | `data/online_books.db` |  freshness_fallback 写入 |
| `user_profiles` | `data/user_profiles.json` | 用户收藏、评分、状态 |
| `item_similarity` | recall SQLite | ItemCF 相似度矩阵 |
| `user_history` | recall SQLite | 用户历史（召回用） |
| ChromaDB | `data/chroma_db` | 向量检索（Dense） |
| ChromaDB Chunks | `data/chroma_chunks` | Small-to-Big 句子级索引 |

---

## 4. 组件依赖图

```mermaid
flowchart TB
    subgraph API["FastAPI 入口"]
        main[main.py]
    end

    subgraph RAG["RAG 路径"]
        orch[RecommendationOrchestrator]
        router[Router]
        vdb[VectorDB]
        reranker[Reranker]
        enricher[MetadataEnricher]
        orch --> router
        orch --> vdb
        orch --> reranker
        orch --> enricher
    end

    subgraph RecSys["推荐路径"]
        rec_svc[RecommendationService]
        fusion[RecallFusion]
        fe[FeatureEngineer]
        ranker[LGBMRanker]
        div_rerank[DiversityReranker]
        rec_svc --> fusion
        rec_svc --> fe
        rec_svc --> ranker
        rec_svc --> div_rerank
    end

    subgraph Data["数据层"]
        meta[MetadataStore]
        online[OnlineBooksStore]
        profile[ProfileStore]
    end

    main --> orch
    main --> rec_svc
    orch --> meta
    orch --> online
    rec_svc --> meta
    div_rerank --> meta
```

---

## 渲染说明

- **GitHub / GitLab**: 默认支持 Mermaid，直接显示
- **VS Code**: 安装 "Markdown Preview Mermaid Support" 扩展
- **命令行**: 使用 `mmdc` (mermaid-cli) 导出 PNG/SVG
  ```bash
  npm install -g @mermaid-js/mermaid-cli
  mmdc -i docs/architecture/ARCHITECTURE_DIAGRAMS.md -o docs/diagrams/
  ```
