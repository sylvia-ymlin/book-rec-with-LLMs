# 贡献指南 (Contributing Guide)

感谢你对本项目的关注。本文档说明如何参与开发、提交流程与约定。

---

## 一、开发环境

### 1.1 环境搭建

```bash
git clone https://github.com/sylvia-ymlin/book-rec-with-LLMs.git
cd book-rec-with-LLMs
conda env create -f environment.yml
conda activate book-rec
```

### 1.2 首次运行

```bash
python src/init_db.py
python scripts/init_sqlite_db.py
make run
```

详见 [README](README.md) 与 [Build Guide](docs/build_guide.md)。

---

## 二、项目结构

```
├── src/                    # 核心代码
│   ├── api/                # FastAPI 路由
│   ├── core/               # 路由、编排、重排等
│   ├── recall/             # 召回通道（ItemCF、SASRec 等）
│   ├── services/           # 推荐、Chat 服务
│   └── ...
├── scripts/                # 数据与训练脚本
├── config/                 # 路由等配置
├── data/                   # 数据目录（不入库）
└── docs/                   # 文档
    ├── DEVELOPMENT.md      # 开发指南（召回、路由）
    ├── TECHNICAL_REPORT.md # 技术报告
    └── ...
```

---

## 三、开发流程

### 3.1 分支与提交

- 从 `master` 切出新分支：`git checkout -b feature/your-feature`
- 提交信息建议：`feat: 简短描述` 或 `fix: 修复描述`
- 完成后再提 PR 到 `master`

### 3.2 代码风格

- **Python**: 遵循 PEP 8，使用 type hints
- **风格**: 研究原型风格，偏简单、线性，避免过度抽象（见 `.cursor/rules/research-prototype-style.mdc`）
- **路径**: 使用 `pathlib.Path`，不用 `os.path.join`

### 3.3 调试与测试

- 关键路径有针对性测试（`tests/`）
- 新召回通道或路由逻辑建议补充对应测试
- 可用 `make run` 本地验证 API 行为

---

## 四、常见贡献场景

| 场景 | 参考文档 |
|------|----------|
| 添加新召回通道 | [docs/DEVELOPMENT.md § 一](docs/DEVELOPMENT.md#一如何添加新的召回通道) |
| 调整路由规则 / 关键词 | [docs/DEVELOPMENT.md § 二](docs/DEVELOPMENT.md#二如何调整路由规则) |
| 修复 Bug | 先复现，再最小改动修复 |
| 性能优化 | 对比前后指标，在 CHANGELOG 中记录 |

---

## 五、文档与变更

- **CHANGELOG.md**: 用户可见的变更应在此记录
- **docs/DEVELOPMENT.md**: 开发相关扩展或修改流程时同步更新
- **README.md**: 面向使用者的说明，重大功能变更时更新

---

## 六、联系方式

- Issue: 在 GitHub 仓库提 Issue
- 讨论: 可在 Issue 中说明设计与实现思路

---

*项目处于维护模式（v2.6.0 冻结），欢迎小范围改进与修复。*
