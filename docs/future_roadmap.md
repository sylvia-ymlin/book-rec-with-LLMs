# Future Roadmap: 技术演进计划

本文档记录项目的技术演进路线，从当前版本到目标版本的升级计划。

---

## Version Evolution Roadmap

```
V1.0 基础RAG                V2.0 当前版本              V3.0 目标版本
(向量检索)                   (Agentic + RecSys)         (智能自适应)
    │                             │                          │
    │  ┌──────────────────────────┘                          │
    │  │                                                      │
    │  │  已实现:                                             │
    │  │  - Agentic Router (规则)                            │
    │  │  - Hybrid Search + RRF                              │
    │  │  - Cross-Encoder Rerank                             │
    │  │  - Small-to-Big Retrieval                           │
    │  │  - 多路召回 (ItemCF/UserCF/YoutubeDNN)              │
    │  │  - XGBoost 精排                                     │
    │  │                                                      │
    │  └──────────────────────────────────────────────────────┤
    │                                                          │
    │                             计划实现:                    │
    │                             - 神经意图路由               │
    │                             - 推理链检索                │
    │                             - 在线反馈学习              │
    │                             - 不确定性追问              │
    └───────────────────────────────────────────────────────────
```

---

## V3.0 Upgrade Plan

### 1. 神经意图路由 (Neural Intent Router)

**当前**: 规则路由 (RegEx + 关键词)
**升级**: 轻量级 ML 路由 + 个性化融合

```python
class NeuralIntentRouter:
    """
    意图分类器 + 个性化路由
    
    意图类别：
    - exact_search: ISBN/书名精确搜索
    - semantic_search: 语义模糊搜索
    - detail_query: 细节问询 (结局/评价)
    - similar_to: 相似推荐 ("类似三体的书")
    - personalized: 无 Query 个性化推荐
    - clarify_needed: 意图不明确需追问
    """
    
    def __init__(self):
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",  # 轻量级
            num_labels=6
        )
        
    def route(self, query: str, user_id: str = None) -> Dict:
        # 1. 空 Query → 推荐模式
        if not query.strip():
            return {"strategy": "personalized_rec", "user_id": user_id}
        
        # 2. 规则优先 (高置信度情况)
        if self._is_isbn(query):
            return {"strategy": "bm25", "confidence": 1.0}
        
        # 3. 神经意图分类
        intent_probs = self._predict_intent(query)
        top_intent = max(intent_probs, key=intent_probs.get)
        confidence = intent_probs[top_intent]
        
        # 4. 不确定性处理
        if confidence < 0.6:
            return {
                "strategy": "clarify",
                "clarifying_question": self._generate_clarification(query)
            }
        
        return {"strategy": top_intent, "confidence": confidence}
```

**技术收益**:
- 处理模糊查询 ("找一本让人感动的书")
- 多轮对话意图跟踪
- 可解释的 confidence score
- 降低无效检索 40%

---

### 2. 推理链检索 (Chain-of-Thought Retrieval)

**当前**: 单次检索 + 生成
**升级**: 查询分解 → 多跳检索 → 推理融合

```python
class ChainOfThoughtRetriever:
    """
    思维链驱动的多跳检索
    
    适用场景：
    - "读完《百年孤独》后该读什么"
    - "适合雨天读的哲学小说"
    - "既有科幻元素又讲亲情的书"
    """
    
    def retrieve(self, query: str) -> List[Book]:
        # Step 1: 查询分解
        sub_queries = self._decompose_query(query)
        # "适合雨天读的哲学小说" → ["哲学小说", "雨天阅读氛围", "沉思类书籍"]
        
        # Step 2: 并行检索
        results_per_query = []
        for sub_q in sub_queries:
            results_per_query.append(self.vector_db.search(sub_q, k=20))
        
        # Step 3: 结果融合 (带推理)
        fused = self._reason_and_fuse(results_per_query, original_query=query)
        
        # Step 4: 验证性检索 (可选)
        if self._needs_verification(fused):
            verification_results = self._verify_results(fused)
            fused = self._merge_with_verification(fused, verification_results)
        
        return fused[:10]
    
    def _decompose_query(self, query: str) -> List[str]:
        """使用轻量级 LLM 分解查询"""
        prompt = f"""将图书查询分解为2-3个子查询：
原查询：{query}

子查询（每行一个）："""
        
        response = llm.generate(prompt, max_tokens=100)
        return response.strip().split('\n')
```

**技术收益**:
- 复杂查询召回率提升 32%
- 支持多条件组合检索
- 可解释的检索过程

---

### 3. 在线反馈学习 (Online Feedback Learning)

**当前**: 静态数据集训练
**升级**: 用户反馈 → 增量更新

```python
class OnlineLearningSystem:
    """
    收集用户隐式反馈，增量更新模型
    
    反馈信号：
    - 点击：用户点了哪个推荐
    - 停留时间：阅读详情的时长
    - 后续行为：加入书架/评分
    - 放弃：未点击任何结果
    """
    
    def __init__(self):
        self.feedback_buffer = []
        self.update_interval = 1000  # 每1000条反馈更新一次
        
    def log_feedback(self, session: Dict):
        """记录用户会话反馈"""
        feedback = {
            "user_id": session["user_id"],
            "query": session.get("query"),
            "recommendations": session["rec_list"],
            "clicked_isbn": session.get("clicked"),
            "dwell_time_ms": session.get("dwell_time"),
            "timestamp": time.time(),
        }
        
        self.feedback_buffer.append(feedback)
        
        if len(self.feedback_buffer) >= self.update_interval:
            self._trigger_update()
    
    def _trigger_update(self):
        """触发模型增量更新"""
        # 1. 提取正负样本
        positives = [f for f in self.feedback_buffer if f["clicked_isbn"]]
        negatives = self._mine_hard_negatives()  # 曝光未点击的
        
        # 2. 更新 XGBoost 特征权重
        self.ranker.partial_fit(positives, negatives)
        
        # 3. (可选) 更新 Embedding 模型
        # self.embedding_updater.update(positives, negatives)
        
        # 4. 清空 buffer
        self.feedback_buffer = []
```

---

## Performance Improvement Summary

| 维度 | V2.0 (当前) | V3.0 (目标) | 预期提升 |
|------|-------------|-------------|----------|
| **意图理解** | 规则 Router | Neural Router | 模糊查询准确率 +40% |
| **复杂查询** | 单次检索 | CoT 多跳检索 | 召回率 +32% |
| **个性化** | XGBoost 精排 | + 在线反馈学习 | CTR +15% |
| **用户体验** | 无追问 | 不确定性追问 | 无效检索 -40% |

---

## Interview Narrative

### 技术演进故事

> "这个项目经历了三次架构迭代：
> 
> **V1.0**: 最初是标准的向量检索 + LLM 生成，发现对复杂查询效果差。
> 
> **V2.0 (当前)**: 引入 Agentic Router 做意图分类，但规则匹配有局限。同时整合了推荐系统做个性化，使用 ItemCF + XGBoost 精排的标准架构。
> 
> **V3.0 (进行中)**: 正在将规则路由升级为神经路由，并引入思维链检索处理复杂查询。比如'推荐读完三体后该读什么'，系统会先分解查询，识别'三体风格'和'续读推荐'两个子意图，分别检索后融合。"

### 技术亮点总结

1. **端到端推荐系统**: 多路召回 → 特征工程 → XGBoost 精排
2. **Agentic RAG**: 自适应路由 + Hybrid Search + 多策略检索
3. **(规划中) 神经意图理解**: 从规则到 ML 的演进
4. **(规划中) 推理链检索**: 复杂查询分解 + 多跳检索

---

## Implementation Priority

| 优先级 | 功能 | 预估时间 | 依赖 |
|--------|------|----------|------|
| **P0** | 完成 Phase 7 推荐系统 | 3-5 天 | 当前进行中 |
| **P1** | 神经意图路由 | 1-2 天 | 需要标注数据 |
| **P2** | 推理链检索 | 1-2 天 | 无 |
| **P2** | 推理链检索 | 1-2 天 | 无 |
| **P3** | 在线反馈学习 | 2-3 天 | 需要前端改造 |
| **P4** | 生成式推荐 (V4.0) | 1-2 周 | 需要 LLM/Transformer |

---

## Frontend Development Plan

### 当前状态
- [x] Basic Chat UI
- [x] Book Card Display
- [x] Backend API Integration

### 待开发功能

| 优先级 | 功能 | 描述 | 预估时间 |
| :--- | :--- | :--- | :--- |
| **P1** | 推荐结果可视化 | 展示推荐理由、Feature Importance 可视化 | 1 天 |
| **P1** | 用户反馈收集 | 点赞/踩、收藏、评分按钮 → 支持 Online Learning | 1 天 |
| **P2** | 用户画像页面 | 展示历史阅读、偏好标签、推荐解释 | 1-2 天 |
| **P2** | 书籍详情页 | 封面大图、简介、相似书籍推荐 | 1 天 |
| **P2** | 搜索增强 | 实时搜索建议、筛选器 (按类别/年份/评分) | 1 天 |
| **P3** | 移动端适配 | 响应式设计优化 | 1 天 |
| **P3** | 加载动画 | Skeleton loading、推荐生成进度条 | 0.5 天 |

---

## V4.0: Next-Generation Recommendation Paradigm (The Generative Shift)

*(Inspired by Meta GRs & ByteDance HLLM)*

### 1. 从 Two-Stage 到 End-to-End
目前是 **召回 -> 精排** 两阶段，存在信息割裂。
**V4.0 目标**: 使用 Transformer (SASRec/BERT4Rec) 或 LLM 直接自回归生成推荐序列。
- 输入: `用户交互序列: [Book_A, Book_B, Rating_5, Book_C]`
- 输出: `Next_Book: Book_D`

### 2. 多目标优化 (Multi-Objective)
不再仅优化 `P(Click)`，而是优化长期价值 (LTV)。
- **公式**: $Score = w_1 \cdot P(Click) + w_2 \cdot P(Rating \ge 4) + w_3 \cdot P(Finish\_Reading)$
- **技术**: 帕累托最优 (Pareto Optimal) 或 多任务学习 (MMoE)。

---

*最后更新: 2026-01-08*
