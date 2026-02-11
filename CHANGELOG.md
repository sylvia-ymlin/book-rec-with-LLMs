# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added - 2024-01-07
- **UI Refinements**: Book detail modal layout improvements
  - Author name displayed separately below book cover
  - Optimized spacing between elements (reduced excessive whitespace)
  - Removed mood/emotion display from detail modal for cleaner interface
  - Review highlights positioned directly after AI highlight box
 - **Summary Quality**: Smarter sentence-based summaries with HTML entity cleanup
   - Prefer Google Books description when available
   - Fallback to dataset description with HTML unescape and sentence truncation

### Added - 2024-01-XX
- **Review Highlights Feature**: Semantic sentence extraction with clustering
  - scripts/extract_review_sentences.py for processing book descriptions
  - Review highlights display in React frontend
  - Average rating display in book detail modal
  - REVIEW_HIGHLIGHTS.md documentation

### Changed - 2024-01-XX
- **Frontend Migration**: Moved from dual UI (Gradio + React) to React-only
  - Updated README.md with React frontend setup instructions
  - Updated Dockerfile to run FastAPI backend (port 8000)
  - Updated docker-compose.yml to remove Gradio service
  - Cleaned up documentation references to Gradio

### Removed - 2024-01-XX
- app.py (264-line Gradio legacy UI)
- Makefile run-ui target
- docker-compose.yml ui service definition

---

### Added - 2024-01-06
- **Real-time Book Cover Fetching**: New `src/cover_fetcher.py` module that fetches book covers dynamically from Google Books API and Open Library
  - LRU cache (1000 items) to avoid redundant API calls
  - Automatic fallback to Open Library if Google Books fails
  - Placeholder images for books without covers
  - ~0.5-1s latency increase per recommendation query (10-20 books)
- **Client-Server Architecture**: Separated UI and API into independent processes
  - API server runs on port 6006 (FastAPI backend)
  - React frontend runs on port 5173 (development)
  - Enables better scalability and deployment flexibility

### Changed - 2024-01-06
- **React Frontend (web/)**: Created modern UI with book search and recommendations
  - React 18 + Vite for fast development
  - Tailwind CSS for styling
  - Book detail modal with review highlights
- **Makefile**: Updated `run` command to explicitly use port 6006 for API server
- **src/recommender.py**: Integrated real-time cover fetcher in `_format_results()`
  - Replaced hardcoded file paths with dynamic API calls
  - Each recommendation now fetches fresh cover URLs
  - Added review_highlights and average_rating fields

### Fixed - 2024-01-06
- Port mismatch between API (8000) and UI (expected 6006)
- API validation errors due to payload field name mismatch
- Response structure improvements for frontend integration

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
- React 18 compatibility issues
- Dockerfile startup command (updated to run FastAPI backend)

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
