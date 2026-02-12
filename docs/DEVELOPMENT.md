# 开发指南 (Development Guide)

面向贡献者与维护者的开发文档，涵盖召回通道扩展、路由规则调整等。

---

## 一、如何添加新的召回通道

召回系统采用 **RRF (Reciprocal Rank Fusion)** 融合多通道结果。每个通道独立产生候选列表，按 `score += weight * 1/(k+rank)` 合并。

### 1.1 召回通道接口约定

每个召回通道需实现：

| 方法 | 用途 | 签名约定 |
|------|------|----------|
| `fit(df)` | 训练/构建模型 | `df` 含 `user_id`, `isbn`, `timestamp` |
| `load()` | 加载已训练模型 | 返回 `bool` 表示是否成功 |
| `recommend(...)` | 生成推荐 | 返回 `List[Tuple[str, float]]`，即 `(isbn, score)` |

`recommend` 的典型签名：

```python
def recommend(self, user_id: str, history_items=None, top_k: int = 100, **kwargs) -> List[Tuple[str, float]]:
    """返回 [(isbn, score), ...] 按 score 降序"""
```

冷启动通道（如 `popularity`）可忽略 `user_id` 和 `history_items`。

### 1.2 添加步骤

**Step 1**: 在 `src/recall/` 下实现新召回器

```python
# src/recall/my_recall.py
from pathlib import Path
from typing import List, Optional, Tuple

class MyRecall:
    def __init__(self, data_dir: str = "data/rec", model_dir: str = "data/model/recall"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # ... 你的模型状态

    def fit(self, df) -> "MyRecall":
        """从 train.csv 训练"""
        # 构建并持久化
        return self

    def load(self) -> bool:
        """加载已训练模型"""
        # ...
        return True

    def recommend(self, user_id: str, history_items=None, top_k: int = 100) -> List[Tuple[str, float]]:
        """返回 [(isbn, score), ...]"""
        # ...
        return [(isbn, score), ...]
```

**Step 2**: 在 `RecallFusion` 中注册

编辑 `src/recall/fusion.py`：

1. 在 `DEFAULT_CHANNEL_CONFIG` 中增加配置：

```python
DEFAULT_CHANNEL_CONFIG = {
    # ... 现有通道
    "my_recall": {"enabled": False, "weight": 1.0},  # 默认关闭，便于 A/B 测试
}
```

2. 在 `__init__` 中实例化：

```python
self.my_recall = MyRecall(data_dir, model_dir)
```

3. 在 `load_models()` 中加载：

```python
self.my_recall.load()
```

4. 在 `get_recall_items()` 中调用：

```python
if cfg.get("my_recall", {}).get("enabled", False):
    recs = self.my_recall.recommend(user_id, history_items, top_k=k)
    self._add_to_candidates(candidates, recs, cfg["my_recall"]["weight"])
```

**Step 3**: 在训练流水线中增加训练阶段

编辑 `scripts/run_pipeline.py` 的 `run_training()`：

```python
from src.recall.my_recall import MyRecall
self._run_step("Train MyRecall", lambda: MyRecall().fit(df))
```

**Step 4**: 启用通道

通过 `channel_config` 覆盖默认配置：

```python
from src.recall.fusion import RecallFusion

fusion = RecallFusion(channel_config={
    "my_recall": {"enabled": True, "weight": 0.8},
})
```

或在 `RecommendationService` / `RecommendationOrchestrator` 传入对应配置。

### 1.3 现有通道一览

| 通道 | 说明 | 默认状态 |
|------|------|----------|
| `itemcf` | 物品协同过滤（时序方向权） | 开启 |
| `sasrec` | 序列模型 | 开启 |
| `youtube_dnn` | 双塔 embedding | 开启 |
| `popularity` | 全局热门（冷启动兜底） | 开启 |
| `usercf` | 用户协同过滤 | 关闭 |
| `swing` | Swing 相似度 | 关闭 |
| `item2vec` | 物品 embedding | 关闭 |

---

## 二、如何调整路由规则

RAG 路径的 `QueryRouter` 决定检索策略：`exact`、`fast`、`deep`、`small_to_big`。

### 2.1 路由流程概览

```
Query → ISBN 检测 → 新鲜度检测 → 意图分类（模型 or 规则） → 决策
```

- **ISBN**：纯正则，优先匹配
- **新鲜度**：检测 "new"、"latest"、年份等，设置 `temporal` 与 `freshness_fallback`
- **意图**：有 `IntentClassifier` 时用模型；否则用规则（关键词）

### 2.2 规则关键词配置

关键词来自 `config/router.json`，可被环境变量覆盖。

**文件结构** (`config/router.json`):

```json
{
  "detail_keywords": ["twist", "ending", "spoiler", "readers", "felt", ...],
  "freshness_keywords": ["new", "newest", "latest", "recent", "modern", ...],
  "strong_freshness_keywords": ["newest", "latest"],
  "natural_language_keywords": ["like", "similar", "recommend", "want", "looking", ...]
}
```

| 字段 | 作用 | 路由结果 |
|------|------|----------|
| `detail_keywords` | 细粒度/剧透类查询 | `small_to_big`（chunk 级检索） |
| `natural_language_keywords` | 推荐/偏好类意图 | `deep`（hybrid + rerank） |
| `freshness_keywords` | 时效性 | `temporal=True` |
| `strong_freshness_keywords` | 强时效 | `freshness_fallback=True`（触发 Web 搜索） |

### 2.3 修改关键词

**方式 1：直接改 `config/router.json`**

```json
{
  "detail_keywords": ["twist", "ending", "spoiler", "your_new_keyword"],
  "natural_language_keywords": ["like", "similar", "recommend", "your_new_keyword"]
}
```

**方式 2：环境变量覆盖**

```bash
export ROUTER_DETAIL_KEYWORDS="twist,ending,spoiler,custom1,custom2"
export ROUTER_CONFIG_PATH="/path/to/custom_router.json"
```

`config.py` 中 `_load_router_config()` 会优先使用 `ROUTER_CONFIG_PATH`，再用 `ROUTER_DETAIL_KEYWORDS` 覆盖 `detail_keywords`。

### 2.4 规则逻辑（fallback）

当 `IntentClassifier` 未加载或失败时，使用 `_route_by_rules()`：

1. 含 `detail_keywords` → `small_to_big`
2. 含 `natural_language_keywords` → `deep`（rerank=True）
3. 词数 ≤ 6 且无 NL 关键词 → `fast`
4. 其它 → `deep`

### 2.5 新增路由策略

若要新增策略（如 `custom`）：

1. 在 `QueryRouter.route()` 中增加分支，返回例如 `strategy: "custom"`
2. 在 `RecommendationOrchestrator._get_recommendations_classic()` 中处理该策略：

```python
if decision["strategy"] == "custom":
    recs = self._custom_retrieval(query, ...)
```

3. 在 `src/core/models.py` 的 `RouterDecision` 中补充文档说明（可选）。

---

## 三、相关文件索引

| 模块 | 路径 |
|------|------|
| 召回融合 | `src/recall/fusion.py` |
| 路由逻辑 | `src/core/router.py` |
| 路由配置加载 | `src/config.py` (`_load_router_config`) |
| 关键词配置 | `config/router.json` |
| 训练流水线 | `scripts/run_pipeline.py` |

---

## 四、调试建议

- **召回**：对单通道单独 `fit()` 后 `recommend()`，验证返回格式与数量
- **路由**：调用 `QueryRouter().route("your query")` 查看决策
- **关键词**：在 `config.py` 中 `print(ROUTER_DETAIL_KEYWORDS)` 等，确认加载正确
