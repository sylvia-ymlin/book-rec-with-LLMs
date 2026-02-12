# 架构改进建议

本文档针对代码审查中提出的四个问题进行逐项分析，并提供渐进式重构路径。

---

## 1. Facade 模式过度：BookRecommender → RecommendationOrchestrator

### 现状

- `BookRecommender` 是纯委托 Facade，所有方法仅转发给 `RecommendationOrchestrator`
- 注释写「为了向后兼容」，但造成两层间接调用
- 调用链：`main.py/benchmark/scripts` → `BookRecommender` → `RecommendationOrchestrator`

### 影响

- 无额外业务价值，仅增加一层调用
- 新进入者易困惑：应直接用 Orchestrator 还是 BookRecommender？
- 维护成本：两处接口需同步更新

### 建议方案

**方案 A：直接使用 Orchestrator（推荐）**

1. 将所有 `BookRecommender` 的引用改为 `RecommendationOrchestrator`
2. 若需保留 `vector_db`、`cache` 等属性访问，在 Orchestrator 中已有，可直接暴露
3. 删除 `src/recommender.py` 中的 `BookRecommender` 类

**影响范围（需修改的引用）**：
- `src/main.py`：`recommender = RecommendationOrchestrator()`
- `benchmarks/benchmark.py`
- `scripts/model/evaluate_rag.py`
- `scripts/data/fetch_new_books.py`
- `tests/test_recommender.py`, `test_memory_efficiency.py`, `test_api.py`

**方案 B：若必须保留 Facade**

- 仅在 `main.py` 等入口处保留一层薄 Facade，作为「对外 API 门面」
- 内部脚本、benchmark、测试一律直接使用 `RecommendationOrchestrator`
- 明确文档：`BookRecommender` 仅用于 HTTP API 边界，内部逻辑用 Orchestrator

---

## 2. 单例模式滥用：metadata_store 全局单例

### 现状

```python
# metadata_store.py
metadata_store = MetadataStore()  # 全局单例
```

- 15+ 模块直接 `from src.core.metadata_store import metadata_store`
- SQLite 使用 `check_same_thread=False` 允许多线程，但单连接在多线程下仍有竞争
- 测试时通过 `metadata_store_inst` 注入 mock，但多数模块仍依赖全局实例

### 并发风险

- SQLite 单连接：多线程并发写可能导致 `sqlite3.ProgrammingError` 或锁竞争
- 全局状态：难以做并行测试（多个 test 共享同一 DB 状态）
- 部署扩展：多进程/多实例时无法共享单例

### 建议方案

**渐进式改造**：

1. **短期**：保持 `metadata_store` 全局，但为 `MetadataStore` 增加「可实例化」支持
   - 去掉 `__new__` 单例，改为普通类
   - 在应用入口（如 `main.py` startup）创建 `metadata_store = MetadataStore(DATA_DIR / "books.db")`
   - 通过依赖注入传递：`RecommendationOrchestrator(metadata_store_inst=metadata_store)`

2. **中期**：引入简单的 DI 容器或工厂
   - 在 `main.py` 或 `services/recommend_service.py` 中统一创建并注入
   - 各组件通过构造函数接收 `MetadataStore`，不再 `import metadata_store`

3. **长期**：多线程/高并发时
   - 使用连接池（如 `sqlalchemy` + SQLite，或迁移到 PostgreSQL）
   - 每线程/每请求一个连接，避免共享单连接

**注入路径示例**：
```
main.py startup
  → metadata_store = MetadataStore(db_path)
  → orchestrator = RecommendationOrchestrator(metadata_store_inst=metadata_store)
  → BookIngestion(metadata_store_inst=metadata_store)
  → FallbackProvider(metadata_store_inst=metadata_store)
  → ... 其他已支持 metadata_store_inst 的组件
```

**需修改的模块**（目前直接 import 全局）：
- `vector_db.py`：需从 Orchestrator 或构造函数传入
- `services/recommend_service.py`：已在用，可改为从上层注入
- `ranking/features.py`：延迟加载，需改为注入
- `scripts/data/fetch_new_books.py`, `scripts/model/evaluate.py`：脚本入口创建实例传入

---

## 3. 配置管理混乱

### 现状

| 来源 | 内容 | 示例 |
|------|------|------|
| `src/config.py` | 路径、模型、TOP_K、RERANKER_BACKEND、DEBUG | `DATA_DIR`, `TOP_K_FINAL`, `RERANKER_BACKEND` |
| `config/router.json` | Router 关键词 | `detail_keywords`, `freshness_keywords` |
| `config/data_config.py` | 数据管道路径、训练参数 | `RAW_BOOKS`, `YOUTUBE_DNN_EPOCHS` |
| 环境变量 | 覆盖/扩展 | `USER_DATA_PATH`, `ROUTER_CONFIG_PATH`, `RERANK_CANDIDATES_MAX` |

### 问题

- 重复定义：`PROJECT_ROOT`、`DATA_DIR`、`EMBEDDING_MODEL` 在 `config.py` 与 `data_config.py` 中均有
- 无统一入口：调用方需知道去哪个文件找配置
- 环境变量分散：有的在 config 中读取，有的在业务代码中直接 `os.getenv`

### 建议方案

**方案 A：最小改动 — 统一入口**

1. 新建 `src/config/settings.py`（或扩展现有 `config.py`）：
   - 聚合 `config.py`、`data_config.py` 中的配置
   - 统一从环境变量读取 override（如 `pydantic-settings` 或简单 `os.getenv`）
   - 导出单一 `Settings` 对象或命名空间

2. 保留 `router.json` 作为「可热更新」的 router 配置，但通过 `config.py` 的 `_load_router_config()` 加载，不新增入口

3. 废弃 `config/data_config.py` 中与 `src/config.py` 重复的项，改为从 `src.config` 导入

**方案 B：使用 pydantic-settings（推荐）**

```python
# src/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_root: Path
    data_dir: Path
    top_k_final: int = 10
    reranker_backend: str = "onnx"
    debug: bool = False
    # ... 其他字段，支持 env 自动映射

    class Config:
        env_prefix = "APP_"  # APP_DEBUG=1 等
```

- 类型安全、自动校验、单一真相来源
- 可按环境（dev/prod）加载不同 `.env` 文件

---

## 4. 缺少领域模型：Dict[str, Any] 泛滥

### 现状

- API 层：`BookResponse`、`RecommendationResponse`（Pydantic）✅
- 内部：`get_book_metadata()` → `Dict[str, Any]`，`enrich_and_format()` → `List[Dict[str, Any]]`
- 无 `Book`、`User`、`BookMetadata` 等领域对象

### 影响

- 易出现 `KeyError`（如 `meta["title"]` 若 key 拼写错误）
- IDE 无补全、类型检查无效
- 数据契约不清晰，难以维护

### 建议方案

**阶段 1：引入 TypedDict（低侵入）**

```python
# src/core/models.py
from typing import TypedDict, Optional

class BookMetadata(TypedDict, total=False):
    isbn13: str
    isbn10: Optional[str]
    title: str
    authors: str
    description: Optional[str]
    simple_categories: Optional[str]
    image: Optional[str]
    average_rating: Optional[float]
    publishedDate: Optional[str]
```

- 将 `format_book_response(meta: Dict[str, Any], ...)` 改为 `format_book_response(meta: BookMetadata, ...)`
- 可逐步替换，不强制所有调用方立即改

**阶段 2：Pydantic 领域模型（推荐）**

```python
# src/core/models.py
from pydantic import BaseModel

class BookMetadata(BaseModel):
    isbn13: str
    title: str
    authors: str = ""
    description: str | None = None
    simple_categories: str | None = None
    image: str | None = None
    average_rating: float | None = None
    publishedDate: str | None = None

    class Config:
        extra = "allow"  # 兼容 DB 中多余字段
```

- `MetadataStore.get_book_metadata()` 返回 `BookMetadata | None`
- `enrich_and_format()` 接收 `List[BookMetadata]`，返回 `List[BookResponse]`
- 边界转换：在 DB/JSON 层做 `BookMetadata.model_validate(row)`，在 API 层做 `BookResponse.model_validate(meta)`

**影响范围**：
- `metadata_store.py`：`get_book_metadata` 返回类型
- `metadata_enricher.py`：`enrich_and_format` 签名
- `response_formatter.py`：`format_book_response`
- `recommendation_orchestrator.py`：中间变量类型
- `fallback_provider.py`、`diversity_reranker.py` 等

---

## 实施优先级建议

| 优先级 | 项目 | 工作量 | 收益 | 风险 | 状态 |
|--------|------|--------|------|------|------|
| 1 | 移除 BookRecommender，直接使用 Orchestrator | 低 | 简化调用链，降低认知负担 | 低 | ✅ 已完成 |
| 2 | 引入 BookMetadata 等 TypedDict/Pydantic | 中 | 类型安全、减少 KeyError | 低 | ✅ 已完成 |
| 3 | 配置统一入口（pydantic-settings） | 中 | 可维护性提升 | 低 | ✅ 已完成 |
| 4 | metadata_store 实例化 + DI | 中-高 | 可测试性、并发安全 | 中 | ✅ 已完成 |

**实施完成** (2026-02-12)：所有四项已按优先级完成。详见 CHANGELOG.md。
