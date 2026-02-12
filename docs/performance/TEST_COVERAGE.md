# 测试覆盖改进指南

当前评分: 65/100 → 目标: 85+

## 当前状态

| 测试类型 | 文件 | 覆盖范围 |
|---------|------|----------|
| 单元测试 | test_recommender.py | RecommendationOrchestrator |
| 单元测试 | test_vector_db.py | VectorDB (singleton, search, hybrid, FTS5) |
| 单元测试 | test_metadata_store.py | MetadataStore |
| 单元测试 | **test_recall.py** | fusion RRF, ItemCF, Swing, PopularityRecall, UserCF |
| 集成测试 | **test_api.py** | /health, /recommend, /categories, /metrics, /similar |
| 端到端 | **test_integration.py** | 完整推荐流程、个性化推荐 |
| 性能测试 | test_memory_efficiency.py | 内存占用 |
| 压力测试 | benchmarks/locustfile.py | Locust 负载模拟 |
| RAG评估 | data/rag_golden.csv | Accuracy@K, Recall@K, MRR@K, NDCG@K |

## 已完成改进

1. **test_api.py**  
   - 修复 mock 方式，使用 `patch("src.main.recommender")` 直接 mock  
   - 新增 `test_recommend_mocked`, `test_categories_mocked`, `test_similar_books_mocked`, `test_metrics_endpoint`  
   - conftest 增加 `mock_main_startup` 避免 startup 加载真实模型  

2. **test_recall.py**  
   - `_merge_config`: 配置合并逻辑  
   - `RecallFusion._add_to_candidates`: RRF 打分  
   - `ItemCF.fit`: 合成数据训练  
   - `ItemCF.recommend`: 空历史返回 []  

3. **test_metadata_store.py**  
   - 修复 `PropertyMock` 导入顺序  

## 建议补充

### 1. 召回/排序层单元测试

- [x] `test_recall.py`: Swing, PopularityRecall, UserCF (合成数据)
- [ ] `test_ranking.py`: Reranker 输入输出格式, LGBMRanker (如有)

### 2. 端到端集成测试

- [x] `test_integration.py`: `/recommend` 与 `/api/recommend/personal` 流程 (mock)
- [ ] 使用真实 DB（可选）的完整流程验证  

### 3. 压力测试 (Locust)

已有 `benchmarks/locustfile.py`，用法:

```bash
# 先启动 API
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# 另开终端
pip install locust
locust -f benchmarks/locustfile.py --host=http://localhost:8000

# 打开 http://localhost:8089 控制负载
```

### 4. RAG 评估集

- [x] `data/rag_golden.csv` 已创建（从 example 复制）
- 格式: `query`, `isbn`, `relevance`, `notes`
- 目标: 500+ 人工标注 Query-Book 对

```bash
python scripts/model/evaluate_rag.py --golden data/rag_golden.csv --top_k 10
```

指标: Accuracy@K, Recall@K, MRR@K, NDCG@K

## 运行测试

```bash
# 全部
pytest tests/ -v

# 仅 API + Recall
pytest tests/test_api.py tests/test_recall.py -v

# 排除需真实数据的
pytest tests/ -v -k "not memory"
```
