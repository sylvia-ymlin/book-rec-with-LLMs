# Roadmap: Technical Evolution Plan

This document records the project's technical evolution. **v2.6.0 is frozen** — no new features.

---

## Version Evolution

```
V1.0 Basic RAG              v2.6.0 Frozen              V3.0 Planned (out of scope)
(Vector Search)             (Agentic + RecSys)          (Adaptive Intelligence)
    |                             |                          |
    |  Implemented:               |                          |
    |  - Agentic Router (rules)   |                          |
    |  - Hybrid Search + RRF      |                          |
    |  - Cross-Encoder Rerank     |                          |
    |  - Small-to-Big Retrieval   |                          |
    |  - 7-Channel Recall + RRF   |                          |
    |  - Model Stacking Ranker    |                          |
    |                             |                          |
    |                             Planned:                   |
    |                             - Neural Intent Router     |
    |                             - Chain-of-Thought Retrieval|
    |                             - Online Feedback Learning |
    |                             - Uncertainty Clarification|
```

---

## Current System Status (v2.6.0 — Frozen)

### RAG System
- [x] Query Router (RegEx + Keyword)
- [x] Hybrid Search (BM25 + Dense)
- [x] Cross-Encoder Reranking
- [x] Small-to-Big Retrieval
- [x] Temporal Dynamics
- [x] Context Compression

### Recommendation System
- [x] ItemCF Recall (+ direction weight V2.5)
- [x] UserCF Recall
- [x] Popularity Recall
- [x] YoutubeDNN Two-Tower
- [x] Swing Recall (V2.5)
- [x] SASRec Recall Channel (V2.5)
- [x] Item2Vec Recall (V2.6) — Word2Vec on interaction sequences
- [x] Feature Engineering
- [x] LGBMRanker + Hard Negatives (V2.5, replaced XGBoost)
- [x] Model Stacking (V2.6) — LGB + XGB → LogisticRegression Meta-Learner
- [x] API Integration

### Frontend
- [x] Basic Chat UI
- [x] Book Card Display
- [x] Backend API Integration
- [x] User Profile Page — React Router + Persona/Stats/Rating Distribution/Progress
- [x] My Bookshelf Page — Filter/Sort/Stats/Rating/Status management
- [x] Frontend Refactor — Monolithic App.jsx → React Router SPA (3 pages + 5 components)

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

| 模块 | 当前实现 (V2.6) | 愿景目标 | Gap |
|:---|:---|:---|:---|
| **召回架构** | 7路召回 + RRF ✅ | 3层 L1/L2/L3 | 🟡 中等 |
| **序列模型** | SASRec (feature + recall) | TiSASRec | 🟡 中等 |
| **排序模型** | Model Stacking (LGB+XGB→Meta) ✅ | + Deep Ranker | 🟢 完成 |
| **评估指标** | HR/MRR | 因果 + 长期价值 | 🔴 需新建 |
| **可解释性** | 无 | SHAP + 推荐理由 | 🟡 中等 |

---

## V2.5 RecSys Enhancements (Tianchi) — Completed 2026-01-29

> **Reference**: Tianchi Top 5/5338 solution

### ItemCF Improvements

| Priority | Feature | Description | Status |
|:---|:---|:---|:---|
| **P0** | **Direction Weight** | Forward=1.0, backward=0.7 | ✅ Done |
| P0 | Created Time Weight | `exp(0.8 ** abs(time_i - time_j))` | Already in V2.0 |

### Feature Engineering

| Priority | Feature | Description | Status |
|:---|:---|:---|:---|
| P0 | Last-N Similarity | max/min/mean similarity to last 5 books | ✅ Done (V2.0) |
| P0 | Category Affinity | Is category in user's preferences | ✅ Done (V2.0) |

### Recall Layer

| Priority | Channel | Algorithm | Status |
|:---|:---|:---|:---|
| **P1** | **Swing** | User-pair overlap weighting | ✅ Done (optimized, 35s) |
| **P1** | **SASRec Recall** | Embedding dot-product retrieval | ✅ Done |
| **P2** | **Item2Vec** | Word2Vec on sequences | ✅ Done (V2.6) |

### Ranking Model

| Priority | Enhancement | Description | Status |
|:---|:---|:---|:---|
| **P1** | **LGBMRanker** | LambdaRank (NDCG优化) | ✅ Done |
| **P1** | **Hard Negative Sampling** | Recall results as negatives | ✅ Done |
| **P2** | **Model Stacking** | XGB + LGB → Meta-Learner | ✅ Done (V2.6) |

### V2.5 Results

| Metric | Pre-V2.5 | V2.5 | Improvement |
|:---|:---|:---|:---|
| HR@10 | 0.1380 | **0.2205** | +59.8% |
| MRR@5 | 0.1295 | **0.1584** | +22.3% |

---

## V2.6 Item2Vec + Model Stacking — Completed 2026-01-29

### New Recall Channel

| Priority | Channel | Algorithm | Status |
|:---|:---|:---|:---|
| **P2** | **Item2Vec** | Word2Vec (Skip-gram) on user interaction sequences | ✅ Done |

- **Reference**: Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for Collaborative Filtering", 2016
- **Params**: `vector_size=64, window=5, min_count=3, sg=1 (Skip-gram), epochs=10`
- **Vocabulary**: 44,157 items
- **Training time**: ~48 seconds (index 15s + Word2Vec 7s + similarity matrix 22s)
- **Fusion weight**: 0.8 (between Popularity 0.5 and CF channels 1.0)

### Model Stacking

| Priority | Enhancement | Description | Status |
|:---|:---|:---|:---|
| **P2** | **Model Stacking** | LGBMRanker + XGBClassifier → LogisticRegression Meta-Learner | ✅ Done |

**Architecture**:
```
Level-1: LGBMRanker (LambdaRank scores) + XGBClassifier (binary probabilities)
Level-2: LogisticRegression([lgb_score, xgb_score]) → final probability
Training: 5-Fold GroupKFold CV → Out-of-Fold predictions → Meta-learner
```

**Meta-learner coefficients**: LGB=1.4901 (dominant), XGB=0.0420, intercept=-0.1171

### Recall Channel Weights (V2.6)

| Channel | Weight |
|:---|:---|
| YoutubeDNN | 0.1 |
| ItemCF | 1.0 |
| UserCF | 1.0 |
| Swing | 1.0 |
| SASRec | 1.0 |
| **Item2Vec** | **0.8** |
| Popularity | 0.5 |

### V2.6 Results

| Metric | V2.5 | V2.6 | Improvement |
|:---|:---|:---|:---|
| HR@10 | 0.2205 | **0.4545** | +106.1% |
| MRR@5 | 0.1584 | **0.2893** | +82.6% |

*(n=2000, Leave-Last-Out, title-relaxed matching)*

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

| Dimension | V2.0 | V2.6 (Current) | V3.0 (Target) |
|:---|:---|:---|:---|
| Intent Understanding | Rule Router | Rule Router | Neural Router |
| Complex Queries | Single retrieval | Single retrieval | CoT Multi-hop |
| Ranking Quality | XGBoost (AUC) | **Model Stacking (LGB+XGB→Meta)** ✅ | + Deep Ranker |
| Recall Diversity | 4 channels | **7 channels (+Swing, +SASRec, +Item2Vec)** ✅ | + Faiss |
| Negative Sampling | Random | **Hard Negatives** ✅ | Curriculum Learning |

---

## Tech Debt Management: 3-4-2-1

| 比例 | 领域 | 内容 |
|:---|:---|:---|
| **30%** | 偿还旧债 | 重构召回, 清理冗余 |
| **40%** | 战略创新 | TiSASRec, GNN |
| **20%** | 基础维护 | 稳定性, 监控 |
| **10%** | 探索调研 | RL, 多模态 |

---

*Frozen January 2026 — v2.6.0*
