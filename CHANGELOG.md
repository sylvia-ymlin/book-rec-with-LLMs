# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Performance benchmarking (`benchmarks/benchmark.py`, `benchmarks/results.md`)
- `/benchmark` API endpoint for live performance testing
- Chinese resume descriptions in interview prep document
- Live production benchmark: **0.3-0.4s** backend latency

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
