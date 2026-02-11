# Roadmap: Technical Evolution Plan

This document records the project's technical evolution from current version to target version.

---

## Version Evolution

```
V1.0 Basic RAG              V2.0 Current Version        V3.0 Target Version
(Vector Search)             (Agentic + RecSys)          (Adaptive Intelligence)
    |                             |                          |
    |  Implemented:               |                          |
    |  - Agentic Router (rules)   |                          |
    |  - Hybrid Search + RRF      |                          |
    |  - Cross-Encoder Rerank     |                          |
    |  - Small-to-Big Retrieval   |                          |
    |  - Multi-Channel Recall     |                          |
    |  - XGBoost Ranking          |                          |
    |                             |                          |
    |                             Planned:                   |
    |                             - Neural Intent Router     |
    |                             - Chain-of-Thought Retrieval|
    |                             - Online Feedback Learning |
    |                             - Uncertainty Clarification|
```

---

## Current System Status (V2.0)

### RAG System
- [x] Query Router (RegEx + Keyword)
- [x] Hybrid Search (BM25 + Dense)
- [x] Cross-Encoder Reranking
- [x] Small-to-Big Retrieval
- [x] Temporal Dynamics
- [x] Context Compression

### Recommendation System
- [x] ItemCF Recall
- [x] UserCF Recall
- [x] Popularity Recall
- [x] YoutubeDNN Two-Tower
- [x] Feature Engineering
- [x] XGBoost Ranker
- [x] API Integration

### Frontend
- [x] Basic Chat UI
- [x] Book Card Display
- [x] Backend API Integration
- [ ] User Profile Page
- [ ] My Bookshelf Page

---

## Architecture Vision

> **核心愿景**: 构建一个 **"懂进化、重价值、可解释"** 的图书推荐生态。

### Multi-Level Retrieval Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  L1 层 (实时响应, <20ms)                                     │
│  ├── 双塔向量召回: User Embedding + Faiss IVFPQ             │
│  ├── 内存缓存: 热门书籍 Top1000                              │
│  └── Session-based 协同: 捕捉会话内兴趣突变                   │
├─────────────────────────────────────────────────────────────┤
│  L2 层 (准实时, 50-100ms)                                    │
│  ├── GNN: DGL 异构图 (书籍-用户-标签)                        │
│  ├── 语义召回: Sentence-BERT 长尾冷启动                      │
│  └── TiSASRec: 时序感知序列预测                              │
├─────────────────────────────────────────────────────────────┤
│  L3 层 (离线挖掘, 分钟/小时级)                                │
│  ├── LDA 主题演化: 长期品味迁移                               │
│  └── UCB/ε-Greedy: 打破信息茧房                              │
└─────────────────────────────────────────────────────────────┘
```

### Current vs Vision Gap

| 模块 | 当前实现 | 愿景目标 | Gap |
|:---|:---|:---|:---|
| **召回架构** | 4路召回 + RRF | 3层 L1/L2/L3 | 🟡 中等 |
| **序列模型** | SASRec (无时间) | TiSASRec | 🟡 中等 |
| **排序模型** | XGBoost (AUC) | LGBMRanker (NDCG) | 🟢 易升级 |
| **评估指标** | HR/MRR | 因果 + 长期价值 | 🔴 需新建 |
| **可解释性** | 无 | SHAP + 推荐理由 | 🟡 中等 |

---

## V2.5 RecSys Enhancements (Tianchi)

> **Reference**: Tianchi Top 5/5338 solution

### ItemCF Improvements

| Priority | Feature | Description | Expected Impact |
|:---|:---|:---|:---|
| **P0** | **Direction Weight** | Forward=1.0, backward=0.7 | MRR +2-3% |
| P0 | Created Time Weight | `exp(0.8 ** abs(time_i - time_j))` | Ranking precision |

### Feature Engineering

| Priority | Feature | Description | Expected Impact |
|:---|:---|:---|:---|
| P0 | Last-N Similarity | max/min/mean similarity to last 5 books | MRR +3-5% |
| P0 | Category Affinity | Is category in user's preferences | MRR +2-3% |

### Recall Layer

| Priority | Channel | Algorithm | Purpose |
|:---|:---|:---|:---|
| **P1** | **Swing** | User-pair overlap weighting | Substitute relationships |
| P2 | Item2Vec | Word2Vec on sequences | Sequential patterns |

### Ranking Model

| Priority | Enhancement | Description | Expected Impact |
|:---|:---|:---|:---|
| **P1** | **LGBMRanker** | LambdaRank (NDCG优化) | MRR +3-5% |
| P2 | Model Stacking | XGB + LGB → Meta-Learner | MRR +2-3% |

---

## V3.0 Upgrade Plan

### 1. Neural Intent Router

**Current**: Rule-based routing (RegEx + Keywords)
**Upgrade**: Lightweight ML router with personalization fusion

Intent categories:
- exact_search: ISBN/title exact search
- semantic_search: Semantic fuzzy search
- detail_query: Detail inquiry (ending/reviews)
- similar_to: Similar recommendations
- personalized: No-query personalized
- clarify_needed: Ambiguous intent

### 2. Chain-of-Thought Retrieval

**Current**: Single retrieval + generation
**Upgrade**: Query decomposition -> Multi-hop retrieval -> Reasoning fusion

### 3. Online Feedback Learning

**Current**: Static dataset training
**Upgrade**: User feedback -> Incremental updates

---

## V4.0: Next-Generation Paradigm

### From Two-Stage to End-to-End

Current: Recall -> Ranking (two stages)
V4.0: Transformer auto-regress recommendation sequences

### Multi-Objective Optimization

Formula: `Score = w1 * P(Click) + w2 * P(Rating >= 4) + w3 * P(Finish)`
Tech: Pareto Optimal or Multi-Task Learning (MMoE)

---

## Implementation Priority

### Phase 1: 基础能力升级 (2-3周)

| 任务 | 详情 | 优先级 | 时间估算 |
|:---|:---|:---|:---|
| **ItemCF 方向权重** | 顺序=1.0, 逆序=0.7 | **P0** | 0.5 day |
| **LGBMRanker** | LambdaRank 优化 NDCG | **P1** | 1 day |
| **Swing 召回** | 阿里协同过滤 | **P1** | 1-2 days |

### Phase 2: 架构演进 (4周)

| 任务 | 详情 | 优先级 |
|:---|:---|:---|
| **TiSASRec** | 升级 SASRec, 增加时间编码 | P1 |
| **Faiss 向量召回** | 替换内存 Dot Search | P1 |
| **SHAP 可解释性** | 特征重要性展示 | P1 |

### Phase 3: 价值对齐 (6周)

| 任务 | 详情 | 优先级 |
|:---|:---|:---|
| **DR 去偏** | Doubly Robust 消除曝光偏差 | P2 |
| **推荐理由** | 可解释性展示模块 | P2 |
| **MTL 多任务** | CTR + Completion + Diversity | P2 |

---

## Performance Summary

| Dimension | V2.0 (Current) | V3.0 (Target) | Expected |
|:---|:---|:---|:---|
| Intent Understanding | Rule Router | Neural Router | +40% accuracy |
| Complex Queries | Single retrieval | CoT Multi-hop | +32% recall |
| Ranking Quality | XGBoost | + LGBMRanker | +5-10% MRR |
| Recall Diversity | 5 channels | + Swing + Item2Vec | +15% coverage |

---

## Tech Debt Management: 3-4-2-1

| 比例 | 领域 | 内容 |
|:---|:---|:---|
| **30%** | 偿还旧债 | 重构召回, 清理冗余 |
| **40%** | 战略创新 | TiSASRec, GNN |
| **20%** | 基础维护 | 稳定性, 监控 |
| **10%** | 探索调研 | RL, 多模态 |

---

*Last Updated: January 2026*
