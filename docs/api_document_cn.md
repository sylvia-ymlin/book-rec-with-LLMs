# API 完整文档（中文）

**最后更新**: 2026-02-12
**受众**: 开发者、面试官、技术评审
**版本**: v2.6.0

---

## 📋 目录

1. [API 概览](#api-概览)
2. [核心推荐 API](#核心推荐-api)
3. [聊天 API（RAG）](#聊天-apirag)
4. [用户收藏 API](#用户收藏-api)
5. [个性化推荐 API](#个性化推荐-api)
6. [书籍管理 API](#书籍管理-api)
7. [监控 API](#监控-api)
8. [完整数据流](#完整数据流)

---

## API 概览

### 系统架构

你的项目有 **两条核心路径**：

```
用户请求
    │
    ├─ 有查询词？ → RAG 路径（语义搜索）
    │   示例："帮我找悲伤的科幻小说"
    │   API: /recommend, /chat/completions
    │
    └─ 无查询词？ → RecSys 路径（个性化推荐）
        示例："给我推荐10本书"
        API: /api/recommend/personalized
```

### API 端点总览

| 端点 | 方法 | 功能 | 路径 |
|------|------|------|------|
| **语义推荐** | POST | 根据自然语言查询推荐书籍（RAG） | RAG |
| **个性化推荐** | GET | 基于用户历史的协同过滤推荐 | RecSys |
| **相似书籍** | GET | 基于某本书找相似书籍 | 内容过滤 |
| **流式聊天** | POST | 与书籍对话（RAG + LLM 生成） | RAG |
| **添加收藏** | POST | 将书籍加入收藏夹 | 用户管理 |
| **书籍推荐亮点** | GET | 为用户生成个性化书籍文案 | 营销 |

---

## 核心推荐 API

### 1. `/recommend` - 语义搜索推荐（RAG 路径）

**用途**: 用户输入自然语言查询，系统返回最相关的书籍。

#### 请求

```http
POST /recommend
Content-Type: application/json

{
  "query": "一本关于人工智能的哲学小说",
  "category": "Fiction",
  "user_id": "local",
  "fast": false,
  "use_agentic": false,
  "async_rerank": false
}
```

#### 参数详解

| 参数 | 类型 | 必填 | 说明 | 默认值 |
|------|------|------|------|--------|
| `query` | string | ✅ | 自然语言查询（如"悬疑小说"） | - |
| `category` | string | ❌ | 类别过滤（Fiction/Romance/All） | "All" |
| `user_id` | string | ❌ | 用户标识符（用于个性化） | "local" |
| `fast` | boolean | ❌ | 跳过重排序（延迟 ~150ms） | false |
| `use_agentic` | boolean | ❌ | 使用 LangGraph 工作流（路由→检索→评估→网络回退） | false |
| `async_rerank` | boolean | ❌ | 异步重排序（先返回 RRF 结果，后台重排序） | false |

#### 响应

```json
{
  "recommendations": [
    {
      "isbn": "0140283331",
      "title": "战争与和平",
      "authors": "列夫·托尔斯泰",
      "description": "史诗般的历史小说...",
      "thumbnail": "https://example.com/cover.jpg",
      "caption": "俄国贵族家庭的命运",
      "tags": ["历史", "战争", "爱情"],
      "average_rating": 4.5,
      "explanations": [
        {
          "feature": "title_similarity",
          "contribution": 0.82,
          "direction": "positive"
        }
      ]
    }
  ]
}
```

#### 内部逻辑流程

```
用户查询 "哲学小说"
    │
    ▼
┌─────────────────────────┐
│  QueryRouter           │ ← 意图分类
│  • ISBN? → EXACT       │
│  • 关键词? → FAST      │
│  • 复杂查询? → DEEP    │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  VectorDB.hybrid_search │ ← 混合检索
│  • BM25 稀疏检索 (50条) │
│  • Dense 稠密检索 (50条)│
│  • RRF 融合 → Top 50    │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  Reranker（可选）       │ ← 精排
│  • Cross-Encoder 打分   │
│  • Top 10 精准结果      │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  MetadataStore         │ ← 元数据补充
│  • 添加完整书籍信息     │
│  • 封面、作者、评分等   │
└────────┬────────────────┘
         ▼
    返回结果
```

**关键技术点**：
1. **Hybrid Search（混合检索）**: BM25（关键词匹配）+ Dense Embeddings（语义理解）
2. **RRF Fusion（倒数排名融合）**: 合并两种检索结果，公式：`score = 1/(k + rank)`
3. **Cross-Encoder Reranking（交叉编码器重排序）**: 精准打分，但延迟 ~400ms

---

### 2. `/api/recommend/similar/{isbn}` - 相似书籍推荐

**用途**: 用户点击某本书后，推荐相似的书籍（内容过滤）。

#### 请求

```http
GET /api/recommend/similar/0140283331?k=10&category=Fiction
```

#### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `isbn` | string (路径参数) | 种子书籍的 ISBN |
| `k` | int (查询参数) | 返回数量 (1-50) |
| `category` | string | 类别过滤 |

#### 响应

与 `/recommend` 相同格式。

#### 内部逻辑

```
ISBN "0140283331"
    │
    ▼
┌─────────────────────────┐
│  MetadataStore         │ ← 查找书籍
│  获取标题、描述        │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  VectorDB.search       │ ← 向量相似度
│  用描述做 embedding    │
│  找最近邻 (cosine)     │
└────────┬────────────────┘
         ▼
    返回相似书籍
```

**技术点**:
- 使用 **Sentence-BERT** (`all-MiniLM-L6-v2`) 将书籍描述编码为 384 维向量
- **ChromaDB** 使用 HNSW 索引做快速近似最近邻搜索
- 延迟：~20-100ms

---

## 聊天 API（RAG）

### 3. `/chat/completions` - 流式聊天

**用途**: 用户与某本书对话，LLM 基于书籍内容回答问题。

#### 请求

```http
POST /chat/completions
Content-Type: application/json
X-LLM-Key: sk-xxxxxxx (OpenAI 时需要)

{
  "isbn": "0140283331",
  "query": "这本书的主题是什么？",
  "user_id": "local",
  "provider": "ollama"
}
```

#### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `isbn` | string | 要聊天的书籍 ISBN |
| `query` | string | 用户问题 |
| `user_id` | string | 用户 ID（用于聊天历史） |
| `provider` | string | LLM 提供商："ollama"（本地）或 "openai" |

#### 响应

**流式响应**（Server-Sent Events）：

```
data: 这
data: 本
data: 书
data: 的
data: 主题
data: 是
data: 战争
data: 与
data: 和平
...
```

#### 内部逻辑

```
用户问题："主题是什么？"
    │
    ▼
┌─────────────────────────┐
│  DataRepository        │ ← 获取书籍元数据
│  • 标题、作者、描述    │
│  • 评论摘要            │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  build_persona()       │ ← 构建用户画像
│  • 用户收藏的书籍      │
│  • 生成个性化 prompt   │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  LLMFactory            │ ← 调用 LLM
│  • Ollama: llama3      │
│  • OpenAI: gpt-4       │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  StreamingResponse     │ ← 流式返回
│  • 实时推送 token      │
│  • 前端逐字显示        │
└─────────────────────────┘
```

**关键技术点**：
1. **RAG（检索增强生成）**: 先检索书籍信息，再生成回答
2. **流式响应**: 使用 Server-Sent Events（SSE），用户体验更好
3. **上下文压缩**: 保留最近 2 轮对话，旧对话用 LLM 总结压缩

---

## 个性化推荐 API

### 4. `/api/recommend/personalized` - 协同过滤推荐（RecSys 路径）

**用途**: 基于用户历史行为，推荐个性化书籍（无需查询词）。

#### 请求

```http
GET /api/recommend/personalized?user_id=A3SPTOKDG7WBLN&k=10&intent=general&enable_diversity=true
```

#### 参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `user_id` | string | 用户 ID | - |
| `k` | int | 返回数量 | 10 |
| `intent` | string | 意图："general"/"favorites"/"new" | "general" |
| `enable_diversity` | boolean | 启用多样性重排序 | true |
| `real_time_isbns` | string | 实时会话书籍（逗号分隔） | - |

#### 响应

与 `/recommend` 相同格式。

#### 内部逻辑（这是你项目的核心！）

```
用户 ID "A3SPTOKDG7WBLN"
    │
    ▼
┌──────────────────────────────────────┐
│  RecallFusion（7 通道召回）          │
│  ┌────────────────────────────────┐  │
│  │ 1. ItemCF（方向加权协同过滤）   │  │ → Top 100
│  │    • 用户读 A 后读 B            │  │
│  │    • forward 权重 1.0           │  │
│  │    • backward 权重 0.7          │  │
│  ├────────────────────────────────┤  │
│  │ 2. SASRec（序列 Transformer）   │  │ → Top 100
│  │    • 自注意力机制               │  │
│  │    • 捕捉阅读顺序               │  │
│  ├────────────────────────────────┤  │
│  │ 3. Swing（用户对重叠加权）      │  │ → Top 100
│  │    • 1/(α + |I_u ∩ I_v|)       │  │
│  ├────────────────────────────────┤  │
│  │ 4. Item2Vec（Word2Vec 嵌入）   │  │ → Top 100
│  ├────────────────────────────────┤  │
│  │ 5. YoutubeDNN（双塔神经网络）   │  │ → Top 100
│  ├────────────────────────────────┤  │
│  │ 6. UserCF（用户相似度）         │  │ → Top 100
│  ├────────────────────────────────┤  │
│  │ 7. Popularity（流行度回退）     │  │ → Top 100
│  └────────────────────────────────┘  │
│  RRF 融合 → Top 100 候选              │
└────────┬─────────────────────────────┘
         ▼
┌─────────────────────────┐
│  FeatureEngineer       │ ← 特征工程
│  • 用户统计 (u_cnt...)  │
│  • 物品统计 (i_cnt...)  │
│  • SASRec 分数          │
│  • ItemCF 分数          │
│  • 17 个排序特征        │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  LGBMRanker            │ ← L2R 排序
│  • LambdaRank 目标      │
│  • 直接优化 NDCG        │
│  • 输出排序分数         │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  DiversityReranker     │ ← 多样性重排序
│  • MMR (λ=0.75)         │
│  • 流行度惩罚 (γ=0.3)   │
│  • 每类别最多 3 本      │
└────────┬────────────────┘
         ▼
    返回 Top K
```

**关键技术点**：

1. **7 通道召回策略**：
   - **ItemCF（物品协同过滤）**: 方向加权，捕捉"先读 A 后读 B"模式
   - **SASRec（序列推荐）**: Transformer 自注意力，理解阅读顺序
   - **Swing**: 用户对重叠惩罚，减少流行度偏差
   - **Item2Vec**: Word2Vec 应用于物品序列
   - **YoutubeDNN**: 深度双塔模型（用户塔 + 物品塔）
   - **UserCF（用户协同过滤）**: Jaccard 相似度
   - **Popularity**: 冷启动回退

2. **RRF（倒数排名融合）**：
   ```python
   # 每个通道贡献的分数
   score += weight * (1.0 / (k + rank + 1))
   # k=60, rank 是 0-based 排名
   ```

3. **LGBMRanker（LambdaRank）**：
   - 直接优化 NDCG（归一化折损累计增益）
   - 不是二分类，是 list-wise 排序
   - 训练时使用 hard negative sampling

4. **多样性重排序（MMR）**：
   ```python
   # Maximal Marginal Relevance
   score_new = λ * 相关性分数 - (1-λ) * 与已选书籍的相似度
   # λ=0.75: 75% 相关性，25% 多样性
   ```

**性能指标**：
- **HR@10 = 0.4545** (命中率：Top 10 中包含用户实际点击书籍的概率)
- **MRR@5 = 0.2893** (平均倒数排名)
- **延迟 P95 < 500ms**

---

## 用户收藏 API

### 5. `/favorites/add` - 添加收藏

```http
POST /favorites/add
Content-Type: application/json

{
  "user_id": "local",
  "isbn": "0140283331"
}
```

#### 响应

```json
{
  "status": "ok",
  "favorites_count": 15
}
```

---

### 6. `/favorites/list/{user_id}` - 获取收藏列表

```http
GET /favorites/list/local
```

#### 响应

```json
{
  "favorites": [
    {
      "isbn": "0140283331",
      "title": "战争与和平",
      "author": "列夫·托尔斯泰",
      "img": "cover.jpg",
      "category": "Fiction",
      "rating": 4.5,
      "status": "finished",
      "added_at": "2026-01-15T10:30:00",
      "comment": "史诗般的作品"
    }
  ]
}
```

---

### 7. `/favorites/update` - 更新收藏

```http
PUT /favorites/update
Content-Type: application/json

{
  "user_id": "local",
  "isbn": "0140283331",
  "rating": 5.0,
  "status": "finished",
  "comment": "太精彩了！"
}
```

---

### 8. `/favorites/persona` - 生成用户画像

**用途**: 基于用户收藏的书籍，生成个性化描述。

```http
GET /favorites/persona?user_id=local
```

#### 响应

```json
{
  "persona": {
    "description": "你是一位热爱经典文学的读者，偏好历史小说和哲学作品...",
    "top_genres": ["历史", "哲学", "文学"],
    "reading_stats": {
      "total_books": 25,
      "avg_rating": 4.3,
      "favorite_authors": ["托尔斯泰", "陀思妥耶夫斯基"]
    }
  }
}
```

---

## 书籍管理 API

### 9. `/books/add` - 添加新书

**用途**: 动态添加书籍到数据库和向量索引。

```http
POST /books/add
Content-Type: application/json

{
  "isbn": "1234567890",
  "title": "新书标题",
  "author": "作者名",
  "description": "这是一本关于...",
  "category": "Fiction",
  "thumbnail": "https://example.com/cover.jpg"
}
```

#### 内部逻辑

```
新书信息
    │
    ▼
┌─────────────────────────┐
│  MetadataStore.add()   │ ← SQLite 插入
│  • 保存元数据          │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  VectorDB.add()        │ ← ChromaDB 索引
│  • 描述 → embedding    │
│  • 加入 HNSW 索引      │
└────────┬────────────────┘
         ▼
    返回成功
```

---

### 10. `/api/book/{isbn}/highlights` - 书籍推荐亮点

**用途**: 为用户生成个性化书籍推荐文案（营销功能）。

```http
GET /api/book/0140283331/highlights?user_id=local
```

#### 响应

```json
{
  "isbn": "0140283331",
  "highlights": [
    "这本书与你喜欢的《安娜·卡列尼娜》风格相似",
    "史诗般的战争场面描写",
    "深刻的人性探讨"
  ],
  "personalized_pitch": "基于你对历史小说的喜爱，这本书会让你爱不释手。"
}
```

---

## 监控 API

### 11. `/health` - 健康检查

```http
GET /health
```

#### 响应

```json
{
  "status": "healthy"
}
```

---

### 12. `/metrics` - Prometheus 指标

**用途**: 暴露系统性能指标，供 Prometheus 抓取。

```http
GET /metrics
```

#### 响应（Prometheus 格式）

```
# HELP http_requests_total Total count of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",endpoint="/recommend",status_code="200"} 1234

# HELP http_request_duration_seconds HTTP request latency in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="POST",endpoint="/recommend",le="0.1"} 450
http_request_duration_seconds_bucket{method="POST",endpoint="/recommend",le="0.5"} 1200
http_request_duration_seconds_sum{method="POST",endpoint="/recommend"} 523.4
http_request_duration_seconds_count{method="POST",endpoint="/recommend"} 1234
```

**监控指标**：
- `http_requests_total`: 总请求数（按方法、端点、状态码分组）
- `http_request_duration_seconds`: 请求延迟直方图

---

## 完整数据流

### 场景 1: 用户搜索 "悲伤的科幻小说"

```
1. 前端发送请求
   POST /recommend
   { "query": "悲伤的科幻小说", "category": "All" }

2. FastAPI 接收 → RecommendationOrchestrator

3. QueryRouter 分类
   → "悲伤的科幻小说" 是复杂查询
   → 策略: DEEP (Hybrid + Rerank)

4. VectorDB.hybrid_search()
   • BM25: "悲伤" OR "科幻" OR "小说" → Top 50
   • Dense: embedding("悲伤的科幻小说") → Top 50
   • RRF 融合 → Top 50

5. Reranker.rerank()
   • Cross-Encoder 打分每个候选
   • 排序 → Top 10

6. MetadataStore.get_books_batch()
   • 补充完整元数据（封面、作者、评分）

7. 返回结果
   {
     "recommendations": [
       {"isbn": "xxx", "title": "三体", ...},
       {"isbn": "yyy", "title": "银河系漫游指南", ...}
     ]
   }
```

**延迟分析**：
- Router: <5ms
- Hybrid Search: ~230ms (P50)
- Reranking: ~400ms
- Metadata: ~10ms
- **总计**: ~650ms (P50), ~1250ms (P95)

---

### 场景 2: 用户请求个性化推荐

```
1. 前端发送请求
   GET /api/recommend/personalized?user_id=A3SPTOKDG7WBLN&k=10

2. FastAPI → RecommendationService

3. RecallFusion.get_recall_items()
   • ItemCF: 用户历史 → 相似物品 → Top 100
   • SASRec: 用户序列 → Transformer → Top 100
   • Swing: 用户对重叠 → Top 100
   • ... (7 通道)
   • RRF 融合 → Top 100 候选

4. FeatureEngineer.extract_features()
   • 对每个候选，提取 17 个特征
   • user_stats: u_cnt, u_mean, u_std
   • item_stats: i_cnt, i_mean, i_std
   • sasrec_score: SASRec 嵌入相似度
   • icf_max: ItemCF 最大分数
   • ...

5. LGBMRanker.predict()
   • 输入: 100 个候选 × 17 特征
   • 输出: 100 个排序分数
   • 排序 → Top 10

6. DiversityReranker.rerank()
   • MMR 多样性重排序
   • 每类别最多 3 本
   • 流行度惩罚

7. MetadataStore.get_books_batch()
   • 补充元数据

8. 返回结果
```

**延迟分析**：
- Recall (7 通道): ~150ms
- Feature Engineering: ~50ms
- LGBMRanker: ~30ms
- Diversity Rerank: ~10ms
- Metadata: ~10ms
- **总计**: ~250ms (P50), ~420ms (P95)

---

### 场景 3: 用户与书籍对话

```
1. 前端发送请求
   POST /chat/completions
   {
     "isbn": "0140283331",
     "query": "这本书的主题是什么？",
     "provider": "ollama"
   }

2. ChatService.chat_stream()

3. DataRepository.get_book_metadata()
   • 获取书籍信息（标题、描述、评论摘要）

4. build_persona()
   • 获取用户收藏的书籍
   • 生成用户画像

5. 构建 Prompt
   ---
   系统: 你是一位文学评论家。

   书籍信息:
   - 标题: 战争与和平
   - 作者: 列夫·托尔斯泰
   - 描述: 史诗般的历史小说...

   用户画像:
   - 喜欢: 历史小说、哲学作品

   用户问题: 这本书的主题是什么？
   ---

6. LLMFactory.create("ollama")
   • 调用 Ollama llama3 模型

7. StreamingResponse
   • 实时推送生成的 token
   • 前端逐字显示

8. 保存聊天历史
   • 用户消息 + AI 回复
   • 最多保留 10 条（5 轮对话）
```

**延迟分析**：
- Metadata: ~10ms
- Persona 构建: ~20ms
- LLM 首 token: ~200-500ms
- 后续 token: ~50ms/token
- **总计**: 流式响应，用户体验好

---

## 外部 API 集成

### Google Books API（新鲜度回退）

**用途**: 当本地数据库没有某本书时，从 Google Books 获取。

```python
# src/core/web_search.py
def fetch_from_google_books(isbn):
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    response = requests.get(url)
    data = response.json()

    # 解析 JSON，提取书籍信息
    book = {
        "isbn": isbn,
        "title": data["items"][0]["volumeInfo"]["title"],
        "authors": data["items"][0]["volumeInfo"]["authors"],
        ...
    }

    # 保存到 online_books.db（暂存库）
    online_books_store.add_book(book)

    return book
```

**何时触发**：
- `/recommend` 请求中包含 "latest" "new" "2024" 等关键词
- `freshness_fallback=True`
- 本地结果不足

---

## 内部 API 架构

### 依赖关系

```
FastAPI main.py
    │
    ├─ /recommend → RecommendationOrchestrator
    │                    ├─ QueryRouter
    │                    ├─ VectorDB
    │                    ├─ Reranker
    │                    └─ MetadataStore
    │
    ├─ /api/recommend/personalized → RecommendationService
    │                                      ├─ RecallFusion
    │                                      │     ├─ ItemCF
    │                                      │     ├─ SASRec
    │                                      │     ├─ Swing
    │                                      │     └─ ...
    │                                      ├─ FeatureEngineer
    │                                      ├─ LGBMRanker
    │                                      └─ DiversityReranker
    │
    └─ /chat/completions → ChatService
                                ├─ DataRepository
                                ├─ build_persona()
                                └─ LLMFactory
```

---

## API 性能基准（v2.6.0）

| 端点 | P50 延迟 | P95 延迟 | 吞吐量 |
|------|---------|---------|--------|
| `/recommend` (FAST) | 230ms | 310ms | 3-4 QPS |
| `/recommend` (DEEP) | 710ms | 1250ms | 1-2 QPS |
| `/api/recommend/personalized` | 245ms | 420ms | 3-5 QPS |
| `/api/recommend/similar/{isbn}` | 50ms | 100ms | 10-15 QPS |
| `/chat/completions` (首 token) | 300ms | 600ms | - |
| `/health` | 5ms | 15ms | 100+ QPS |

---

## 总结

### 核心技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **API 框架** | FastAPI | 异步 API，自动生成 OpenAPI 文档 |
| **向量数据库** | ChromaDB | 存储书籍描述的 384 维 embedding |
| **稀疏检索** | BM25Okapi | 关键词精确匹配 |
| **重排序** | Cross-Encoder (ms-marco) | 精准排序 |
| **元数据存储** | SQLite (FTS5) | 零 RAM 模式，全文搜索 |
| **LLM** | Ollama (llama3) / OpenAI | 聊天生成 |
| **排序模型** | LightGBM (LambdaRank) | 优化 NDCG |
| **序列模型** | SASRec (PyTorch) | Transformer 序列推荐 |
| **监控** | Prometheus | 性能指标 |

### 你的项目亮点

1. **双路径架构**: RAG（查询）+ RecSys（个性化）
2. **7 通道召回**: 协同过滤 + 深度学习 + 流行度
3. **方向加权 ItemCF**: 创新点，捕捉阅读顺序
4. **LambdaRank 优化**: 直接优化 NDCG，不是二分类
5. **流式 LLM 响应**: SSE 实时推送，用户体验佳
6. **生产级监控**: Prometheus 指标，可观测性强

---

**最后更新**: 2026-02-12
**维护者**: 项目团队
