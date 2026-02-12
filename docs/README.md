# Project Documentation Hub

This file is the canonical navigation entry for project documentation.

## Current Status

- **Baseline**: v2.6.0 is frozen for portfolio use.
- **Allowed updates**: bug fixes, engineering refactor, documentation cleanup.
- **Not in scope**: new model/features or new experiment tracks.

## 1) Start Here (Most readers)

| Document | Audience | Purpose |
|:---|:---|:---|
| [../README.md](../README.md) | Recruiter / first-time visitor | 5-minute project overview and quick start |
| [architecture/TECHNICAL_REPORT.md](architecture/TECHNICAL_REPORT.md) | Engineer / interviewer | Full architecture and design decisions |
| [experiments/experiment_archive.md](experiments/experiment_archive.md) | Research / reviewer | Consolidated experiment timeline and conclusions |
| [../CHANGELOG.md](../CHANGELOG.md) | Maintainer | Versioned change history |

## 2) Engineering Guides (Contributors)

| Document | Purpose |
|:---|:---|
| [development/DEVELOPMENT.md](development/DEVELOPMENT.md) | Extend recall/ranking/router and maintain pipeline |
| [development/build_guide.md](development/build_guide.md) | Local build and service startup pipeline |
| [development/huggingface_deployment.md](development/huggingface_deployment.md) | HF Spaces deployment notes |
| [performance/memory_optimization.md](performance/memory_optimization.md) | Zero-RAM SQLite and memory trade-offs |
| [performance/LATENCY_OPTIMIZATION.md](performance/LATENCY_OPTIMIZATION.md) | Latency tuning options and trade-offs |
| [performance/performance_debugging_report.md](performance/performance_debugging_report.md) | Root-cause analysis and debugging playbook |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guidelines |

## 3) Presentation Material

| Document | Purpose |
|:---|:---|
| [presentation/interview_guide.md](presentation/interview_guide.md) | Interview Q&A and STAR examples |
| [presentation/roadmap.md](presentation/roadmap.md) | Technical evolution and vision gap |

## 4) Archives and Raw Reports

| Path | Purpose |
|:---|:---|
| [archived/README.md](archived/README.md) | Archived document index with reason tags |
| [archived/](archived/) | Deprecated or superseded docs |
| [experiments/reports/](experiments/reports/) | Raw experiment reports |
| [../data/legacy_root_exports/](../data/legacy_root_exports/) | Historical root-level data snapshots moved out of repo root |

## Documentation Rules

Each active document should include a short metadata block near the top:

- `Status`: `active` | `frozen` | `deprecated` | `archived`
- `Audience`: target reader
- `Last Updated`: `YYYY-MM-DD`
- `Owner`: maintainer name

When code behavior changes, update:

1. `CHANGELOG.md`
2. One topic document in `docs/`
3. This file (`docs/README.md`) if navigation changed

---

**Frozen baseline metrics (v2.6.0):** HR@10 = 0.4545, MRR@5 = 0.2893
