# Latency Optimization: Full Recommendation Pipeline

## Current State

| Metric | Value | Target (Spotify-style) |
|--------|-------|------------------------|
| P95 Full Recommendation | ~1250ms | < 100ms |
| Mean | ~700–900ms | - |

**面试官点评**: 在 Spotify，推荐接口通常要在 100ms 内返回。1.2s 对用户来说是可以感知的卡顿。

---

## Latency Breakdown

Approximate warm-query breakdown (from `benchmarks/benchmark.py` + `docs/experiments/reports/rerank_report.md`):

| Stage | Location | Latency | Notes |
|-------|----------|---------|-------|
| Router | `src/core/router.py` | ~1ms | Rule-based, fast |
| Sparse (FTS5) | `vector_db._sparse_fts_search` | ~20–50ms | SQLite MATCH |
| Dense (Chroma) | `vector_db.search` | ~50–100ms | HNSW + MiniLM |
| RRF Fusion | `vector_db.hybrid_search` | ~5ms | In-memory |
| **Cross-Encoder Rerank** | `src/core/reranker.py` | **~400–900ms** | **主要瓶颈** |
| Metadata Enrichment | `enrich_and_format` | ~50–100ms | SQLite lookups |

**Rerank 详情**:
- 模型: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- 候选数: `max(k*4, 20)` = 50 (当 `k=10`)
- 每个 (query, doc) pair 需完整前向传播
- 50 对 × ~15–20ms/pair ≈ 750–1000ms

---

## Root Causes

1. **Cross-Encoder 过重**
   - 每对 (query, doc) 都要做完整 attention，无法像 Bi-Encoder 那样预计算 doc 向量
   - 候选数 50 导致串行推理时间长

2. **Benchmark 查询全部触发 Rerank**
   - `TEST_QUERIES` 均为自然语言（如 "a romantic comedy set in New York"）
   - Router 规则: `len(words) > 2` 且无 detail 关键词 → **DEEP** → `rerank=True`
   - 所以每次 benchmark 都跑 Cross-Encoder

3. **LangGraph Agentic 模式更慢**
   - Router → Retrieve → Evaluate（LLM 调用）→ 可选 Web Fallback
   - 串行执行，无并行优化

---

## Optimization Options

### 1. 裁剪候选集（Quick Win）

**当前**: `rerank_candidates = top_candidates[:max(k*4, 20)]` → 50 个

**建议**: 降为 20 个，或通过 config 可配置

```python
# config.py
RERANK_CANDIDATES_MAX = 20  # 从 50 降到 20，预期 latency 减半
```

**Trade-off**: 若 Top-20 中漏掉真实相关书，召回会略降；通常 20 足够覆盖。

---

### 2. ColBERT（Late Interaction）替代 Cross-Encoder

**原理**: ColBERT 对 query 和 doc 分别编码，再用 token-level MaxSim 打分，doc 向量可预计算缓存。

| 方案 | 推理方式 | 预计算 | 典型 Latency |
|------|----------|--------|--------------|
| Cross-Encoder | 每对 (q,d) 完整 forward | 否 | ~15–20ms/pair |
| ColBERT | q 编码 1 次 + doc 向量 dot | 是（doc 可缓存） | ~2–5ms/doc |

**实现要点**:
- 使用 `colbert-ai/colbertv2` 或类似库
- 预计算书籍描述的 token embeddings 存入向量库
- 在线只需 encode query + 与候选 doc 向量做 MaxSim

**Trade-off**: 需要额外索引建设和依赖，效果可能与 Cross-Encoder 相当或略逊。

---

### 3. Rerank 异步化

**思路**: 先返回 Hybrid RRF 的 Top-K，再后台异步 Rerank，结果通过 WebSocket/轮询或下次请求返回。

```
用户请求 → 立即返回 RRF Top-10 (~150ms) → 后台 Rerank → 推送精排结果（可选）
```

**Trade-off**: 实现复杂，需改动 API 和前端；首屏结果质量略降。

---

### 4. ONNX 量化（已有规划）

`rerank_report.md` 已提到: 使用 Cross-Encoder 的 ONNX 版本可获约 2x 加速。

---

### 5. 动态 Rerank 策略（已部分实现）

Router 已对 ISBN/关键词 禁用 Rerank；可进一步收紧：
- 仅当 query 长度 > 某阈值且非纯关键词时启用
- 或增加「低延迟模式」：用户可选「快速」vs「精准」

---

## Implementation Status (v2.7+)

| 优化 | 状态 | 说明 |
|------|------|------|
| 1. 裁剪候选集 | ✅ | `RERANK_CANDIDATES_MAX=20` (config), env 可覆盖 |
| 2. ColBERT | ✅ | `RERANKER_BACKEND=colbert`, 需 `pip install llama-index-postprocessor-colbert-rerank` |
| 3. Rerank 异步化 | ✅ | `fast=true` 跳过 rerank; `async_rerank=true` 先返 RRF，后台精排并缓存 |
| 4. ONNX 量化 | ✅ | `RERANKER_BACKEND=onnx` (默认), 需 `onnxruntime` |

### API 用法

```bash
# 快速模式 (~150ms)
curl -X POST /recommend -d '{"query":"romantic comedy","fast":true}'

# 异步精排：先返 RRF，下次同 query 返缓存精排
curl -X POST /recommend -d '{"query":"romantic comedy","async_rerank":true}'
```
