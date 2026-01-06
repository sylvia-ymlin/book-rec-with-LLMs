# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Super App Architecture**: Transformed into "End-to-End AI E-Commerce Platform" with 3-tab UI.
- **Data**: Integrated Amazon Books 200k Dataset.
- **Features**:
  - Discovery Tab (Redis + ChromaDB).
  - Assistant Tab (RAG Shopping Agent).
  - Marketing Tab (Content Gen + Guardrails).
- **Benchmarks**: Added `/benchmark` endpoint (0.3s latency).
- **CI**: Added GitHub Actions workflow (`ci.yml`).

### Changed
- **Docs**: Renamed `INTERVIEW_PREP.md` to `interview_prep.md` and updated to academic style.


### Changed
- Reorganized project structure: `data/`, `assets/`, `notebooks/` directories
- Updated `src/config.py` with new data paths
- Updated README with project structure section

### Fixed
- Gradio 6.0 compatibility (removed `gr.Div`, simplified theme)
- Dockerfile startup command (FastAPI → Gradio for HF Spaces)

---

## [1.0.0] - 2025-10-20

### Added
- Initial release
- Semantic search using Sentence Transformers (MiniLM-L6-v2)
- ChromaDB vector database with HNSW indexing
- Emotion classification (DistilRoBERTa + GoEmotions)
- Zero-shot genre classification (BART-MNLI)
- FastAPI backend service
- Gradio frontend UI
- Docker containerization
- Hugging Face Spaces deployment
- Data exploration notebooks
