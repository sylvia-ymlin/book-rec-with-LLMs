# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added - 2026-01-06
- **Real-time Book Cover Fetching**: New `src/cover_fetcher.py` module that fetches book covers dynamically from Google Books API and Open Library
  - LRU cache (1000 items) to avoid redundant API calls
  - Automatic fallback to Open Library if Google Books fails
  - Placeholder images for books without covers
  - ~0.5-1s latency increase per recommendation query (10-20 books)
- **Client-Server Architecture**: Separated UI and API into independent processes
  - API server runs on port 6006 (FastAPI backend)
  - UI runs on port 7860 (Gradio frontend)
  - Enables better scalability and deployment flexibility

### Changed - 2026-01-06
- **app.py**: Refactored to use REST API calls instead of direct model loading
  - Removed local model initialization to reduce memory footprint
  - Added proper error handling for API communication
  - Fixed Gradio 6.0 compatibility (moved theme to launch method, added allowed_paths)
  - Fixed payload format to match API schema (query, category, tone)
- **Makefile**: Updated `run` command to explicitly use port 6006 for API server
- **src/recommender.py**: Integrated real-time cover fetcher in `_format_results()`
  - Replaced hardcoded file paths with dynamic API calls
  - Each recommendation now fetches fresh cover URLs

### Fixed - 2026-01-06
- Port mismatch between API (8000) and UI (expected 6006)
- Gradio InvalidPathError for local file paths from old project directory
- API validation errors due to payload field name mismatch (description vs query)
- Response structure mismatch (direct list vs {recommendations: []} object)

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
