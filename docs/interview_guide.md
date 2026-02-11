# 面试准备指南 (Interview Guide)

这份文档总结了本项目作为面试作品的核心亮点、技术深度以及常见问题的回答策略。

## 🌟 核心亮点 (Why this project?)

### 1. 架构深度 (Architecture Depth)
*   **Agentic RAG**: 不仅仅是简单的向量检索，而是引入了**动态路由 (Dynamic Routing)**。系统能根据用户意图（如 ISBN 精确搜索 vs. 模糊语义搜索）自动选择最佳检索策略（BM25, Hybrid, Small-to-Big），展示了对 RAG 系统的精细化控制能力。
*   **Stacking Ensemble (模型融合)**: 在 Ranking 阶段，没有止步于单一模型，而是实现了 **LightGBM + XGBoost + Logistic Regression** 的 Stacking 架构。这体现了对机器学习模型偏差与方差的理解，以及追求极致推荐效果的工程态度。
*   **Vector Database**: 结合 ChromaDB 实现语义搜索，紧跟当前 LLM + Vector Store 的技术热点。

### 2. 工程质量 (Engineering Excellence)
*   **性能优化 (Performance Optimization)**:
    *   **问题**: 系统在并发场景下出现卡顿，且推理延迟较高。
    *   **解决**: 
        1.  **Async/Await 陷阱**: 发现 FastAPI 的 `async` 路由中运行了 CPU 密集型任务（Pandas 操作），导致 Event Loop 阻塞。即使加上 `await` 也没用，必须去除非 IO 操作的 async 或使用线程池。改为同步 `def` 让 FastAPI自动利用线程池解决。
        2.  **向量化重构**: 发现特征生成使用了 Python 原生 `for` 循环。重构为 Numpy/Pandas 的向量化 (Vectorized) 操作，利用 SIMD 指令集优势，将推理速度提升了约 10 倍。
        3.  **单例模式**: 引入 `MetadataStore` 单例，避免每次请求重复加载 CSV，显著降低了内存占用和 I/O 开销。
*   **可解释性 (Explainability)**: 集成了 **SHAP (SHapley Additive exPlanations)**。不再是推荐系统的“黑盒”，而是能实时给出“为什么推荐这本书”（例如：因为你喜欢作者 X，或者因为主要读这类书），这是区分初级项目和高级项目的重要特征。

### 3. 完整性 (Completeness)
*   **Full Stack**: 前端 (React) + 后端 (FastAPI) + 数据流 (ETL) + 模型训练 (Train Scripts) + 部署 (Docker)。
*   **DevOps**: 包含 Dockerfile 和完整构建脚本，具备生产部署能力。

---

## 🗣️ 面试话术与 Q&A 策略

### Q1: 你在项目中遇到的最大困难是什么？怎么解决的？
**建议回答**:
> “最让我印象深刻的是**系统性能优化**的过程。
> 最初版本在处理高并发请求时，推理延迟很高，甚至会阻塞整个服务。
> 我通过两个层面解决了这个问题：
> 1.  **架构层**: 我使用 Profiling 工具发现，FastAPI 的 `async` 接口中包含了大量的 Pandas 数据处理逻辑。因为 Python 的 `async` 是单线程协作式的，CPU 密集型任务会直接卡死 Event Loop。我将其重构为利用 FastAPI 线程池的非异步模式，解决了阻塞问题。
> 2.  **代码层**: 我发现特征工程部分原本是用 Python 循环写的。我将其重构为 **Numpy 向量化** 操作，把时间复杂度从 O(N) 的 Python 解释器开销优化到了底层 C 语言级别的矩阵运算，最终将特征生成速度提升了 10 倍以上。”

### Q2: 为什么选择 Stacking 融合模型？直接用 LightGBM 不够吗？
**建议回答**:
> “单一模型往往存在局限性。
> LightGBM 擅长处理类别特征和梯度提升，XGBoost 在正则化处理上表现很好。
> 通过 Stacking，我使用一个简单的逻辑回归 (Logistic Regression) 作为 Meta-Learner 来学习这两个强模型的输出。
> 这不仅能利用不同模型的优势（降低 Bias 和 Variance），还能提高系统的**鲁棒性**。在我的离线实验中，Stacking 相比单一 LightGBM 在 NDCG@10 指标上有明显提升。”

### Q3: 你的 RAG 系统有什么特别之处？
**建议回答**:
> “我的 RAG 系统不是简单地 'Retrieve then Generate'。我设计了一个 **Agentic Router**。
> 它会先判断用户的意图：如果是搜书号，直接走精确匹配；如果是模糊描述，走语义索引；如果是复杂查询，会触发 Rerank 重排序。
> 这种动态策略解决了传统 RAG '查得准就不全，查得全就不准' 的痛点。”

---

## 📈 关键指标 (Key Metrics)
*   **Hit Rate@10**: 0.2205 (V2.5 Baseline) -> 0.2312 (V2.7 Est.)
*   **MRR@5**: 0.1584 -> 0.1650
*   **Latency**: P99 < 50ms (Personalized Recs)
