# Project Documentation

## Layer 1 — Main Story (README, 5-min interview)

| Document | Purpose |
|:---|:---|
| [Technical Report](TECHNICAL_REPORT.md) | Architecture, design decisions, method line |
| [Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md) | 流程图、时序图、ER 图（Mermaid） |
| [Experiment Archive](experiments/experiment_archive.md) | Consolidated experiment log (V1.0 → v2.6.0) |

## Layer 2 — Capability Showcase (Resume, technical Q&A)

| Document | Purpose |
|:---|:---|
| [Development Guide](DEVELOPMENT.md) | 添加召回通道、调整路由规则 |
| [Contributing](../CONTRIBUTING.md) | 贡献者指南 |
| [Interview Guide](interview_guide.md) | Q&A, STAR cases |
| [Memory Optimization](memory_optimization.md) | Zero-RAM SQLite, engineering decisions |
| [Performance Debugging](performance_debugging_report.md) | Root cause analysis |
| [Build Guide](build_guide.md) | Full build pipeline |
| [Hugging Face Deployment](huggingface_deployment.md) | HF Spaces deployment |

## Archives

| Path | Contents |
|:---|:---|
| [archived/](archived/) | Deprecated docs (Phase 2, TAGS, REVIEW_HIGHLIGHTS, etc.) |
| [archived/graveyard/](archived/graveyard/) | Layer 3 — tried but not in main story (future_roadmap, interview_deep_dive, etc.) |
| [experiments/reports/](experiments/reports/) | Raw experiment reports (baseline, hybrid, rerank, router, temporal) |

---

**Frozen v2.6.0** — HR@10 = 0.4545, MRR@5 = 0.2893
