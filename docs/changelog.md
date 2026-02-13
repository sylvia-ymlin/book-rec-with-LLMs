# Changelog

All notable changes to this project will be documented in this file.

## [2.6.0] - 2026-02-11

### Frozen — Project Freeze
- **Version freeze**: Project locked at v2.6.0 for portfolio use. No new features, experiments, or optimizations. Documentation and bug fixes only.
- **Version alignment**: Updated `src/app/main.py`, `web/package.json` to 2.6.0. Unified version references across README, docs, technical_report, experiment_archive, interview_guide, roadmap.
- **Freeze notice**: Added to README, docs/README, experiment_archive, roadmap.

## [Post-freeze maintenance] - 2026-02-12

Post-freeze updates after v2.6.0 baseline lock. Scope: bug fixes, architecture cleanup, and documentation/engineering maintenance.

### Added - A/B Testing & RAG Diversity (2026-02-12)
- **A/B testing framework** (`src/core/ab_experiments.py`): Minimal experiment assignment via `get_variant(user_id, experiment_id)`; `get_experiment_config()` returns control/treatment params. Enable with `AB_EXPERIMENTS_ENABLED=true`.
- **API params**: `POST /recommend` and `GET /api/recommend/personal` accept `experiment_id`, `ab_variant` ("control"|"treatment"). `diversity_rerank` experiment toggles `enable_diversity_rerank` for online validation.
- **RAG path diversity**: `RecommendationOrchestrator` now applies `DiversityReranker` (MMR + popularity penalty + category constraint) to RAG search results. Configurable via `ENABLE_RAG_DIVERSITY=true` (default).

### Changed - Architecture Improvements (2026-02-12)
- **Removed BookRecommender facade**: Direct use of `RecommendationOrchestrator` throughout (main.py, benchmarks, scripts, tests). Eliminates redundant indirection.
- **BookMetadata domain model**: Added `src/core/models.py` with `BookMetadata` TypedDict for type safety in metadata flow. Updated `metadata_store`, `response_formatter` type hints.
- **Unified config**: `src/infra/config.py` now uses `pydantic-settings` `Settings` class. Single source of truth; env vars override defaults. Added `get_settings()` for typed access.
- **MetadataStore instance + DI**: Removed singleton pattern; `MetadataStore(db_path)` is now a normal class. Supports explicit instantiation for tests. Global `metadata_store` retained for backward compat; main.py explicitly passes to `RecommendationOrchestrator`.

### Fixed - Router heuristic fragility (intent_classifier)
- **Model-based routing**: Trained `intent_classifier.pkl` (TF-IDF + LogisticRegression) with book title examples; router now uses model when available.
- **SEED_DATA extended**: Added `War and Peace`, `The Lord of the Rings`, `Harry Potter`, `1984`, etc. (fast) and `books like War and Peace`, `similar to The Lord of the Rings` (deep) so model distinguishes book titles from recommendation-style queries.
- **Fallback rules improved**: Replaced brittle `len(words) <= 2` with NL keyword detection (`ROUTER_NL_KEYWORDS`: like, similar, recommend, want, looking, ...). Short queries (≤6 words) without NL keywords → FAST; queries with NL keywords → DEEP.
- **Config**: `natural_language_keywords` in router config; `ROUTER_NL_KEYWORDS` in `src/infra/config.py`.

### Added - Latency Optimizations (`docs/performance/LATENCY_OPTIMIZATION.md`)
- **1. 裁剪候选集**: `RERANK_CANDIDATES_MAX=20` (env overridable); rerank top 20 instead of 50.
- **2. ColBERT**: `RERANKER_BACKEND=colbert`; optional `llama-index-postprocessor-colbert-rerank`.
- **3. Rerank 异步化**: `fast=true` skips rerank (~150ms); `async_rerank=true` returns RRF first, reranks in background, next request gets cached reranked.
- **4. ONNX 量化**: `RERANKER_BACKEND=onnx` (default); `onnxruntime` for ~2x CrossEncoder speedup.
- API: `POST /recommend` accepts `fast`, `async_rerank`; `web/src/api.js` updated.

### Added - Cold-Start Optimizations (P0–P2)
- **P0**: Popularity fallback — enabled Popularity channel by default; `RecallFusion` and `RecommendationService` fallback to popular books when all recall channels return empty.
- **P0**: `recent_isbns` API param — `/api/recommend/personal` accepts comma-separated ISBNs from current session; injected into SASRec for 1-click cold-start convergence.
- **P1**: Frontend passes `recent_isbns` — session-level tracking of viewed books; passed to personalized API on Start Discovery.
- **P2**: Onboarding flow — `OnboardingModal` when new user (no collection); pick 3–5 books from popular list to seed preferences; `GET /api/onboarding/books`.
- **P2**: Zero-shot intent probing — `src/rag/intent_prober.py` uses LLM to infer categories/emotions/keywords from user query; `GET /api/intent/probe`; `intent_query` param on personal API seeds SASRec via semantic search when user has no history.

### Added - 2026-01-29 (Frontend Refactor: React Router SPA)
- **React Router SPA**: Refactored monolithic 960-line `App.jsx` into React Router architecture with 3 route pages and 5 reusable components.
  - Routes: `/` (Gallery), `/bookshelf` (My Bookshelf), `/profile` (User Profile)
  - Components: `Header`, `BookCard`, `BookDetailModal`, `SettingsModal`, `AddBookModal`
  - Pages: `GalleryPage`, `BookshelfPage`, `ProfilePage`
- **User Profile Page** (NEW): Displays AI-generated reading persona, stats overview (total books, completion rate, avg rating, currently reading), favorite authors & top categories from backend persona API, rating distribution bar chart, reading progress visualization, and recently finished books.
- **My Bookshelf Page**: Dedicated page with filter (all/want_to_read/reading/finished), sort (recent/rating/title), statistics cards, and mood preference display.
- **Dependencies**: Added `react-router-dom` for client-side routing.

### Added - 2026-01-29 (V2.6 Item2Vec + Model Stacking)
- **Item2Vec Recall Channel**: Word2Vec (Skip-gram) trained on user interaction sequences to learn item embeddings (`src/recsys/recall/item2vec.py`). 44,157 items in vocabulary, cosine similarity matrix for fast retrieval. Added as 7th recall channel with weight=0.8.
- **Model Stacking Ranker**: Two-level ensemble — Level-1: LGBMRanker (LambdaRank) + XGBClassifier (binary logistic), Level-2: LogisticRegression meta-learner trained on 5-Fold GroupKFold out-of-fold predictions. Backward compatible — falls back to LGB-only if stacking files absent.
- **Dependencies**: Added `gensim>=4.3.0` and `xgboost>=2.0.0` to requirements.
- **Results**: HR@10 improved from 0.2205 to **0.4545** (+106.1%), MRR@5 from 0.1584 to **0.2893** (+82.6%) on n=2000 evaluation.

### Added - 2026-01-29 (V2.5 RecSys Enhancements)
- **Swing Recall Channel**: New collaborative filtering algorithm based on user-pair overlap weighting (`src/recsys/recall/swing.py`). Optimized from O(items × users²) to O(users × items_per_user²) — trains in 35 sec instead of 2+ hours.
- **SASRec Recall Channel**: Dot-product retrieval using pre-computed SASRec embeddings (`src/recsys/recall/sasrec_recall.py`). Now serves as both a ranking feature and an independent recall source.
- **Hard Negative Sampling**: Ranker training mines negatives from recall results instead of random items, teaching the model to distinguish "close but wrong" from "correct".
- **LGBMRanker (LambdaRank)**: Replaced XGBoost binary classifier with LightGBM LambdaRank that directly optimizes NDCG.
- **ItemCF Direction Weight**: Asymmetric similarity — forward co-occurrence (item1 read before item2) weighted 1.0, backward 0.7.
- **Results**: HR@10 improved from 0.1380 to **0.2205** (+59.8%), MRR@5 from 0.1295 to **0.1584** (+22.3%) on n=2000 evaluation.

### Fixed - 2026-01-29 (Performance Optimization)
- **Restored Recommendation Performance**: Improved **Hit Rate@10** from 0.012 to **0.138** and **MRR@5** to **0.129**.
- **Recall Fusion Tuning**: Reduced `YoutubeDNN` weight (2.0 -> 0.1) to prevent high-bias results from burying ItemCF/Swing collaborative signals.
- **Evaluation Pipeline**:
  - Implemented **Title-Based Evaluation** to correctly handle hits where a different edition (ISBN) of the target book is recommended.
  - Added `filter_favorites` toggle to `get_recommendations` to bypass data leakage during evaluation.
- **Deduplication Logic**: Refactored `RecommendationService` to correctly handle title collisions without dropping high-ranked items.

### Added - 2026-01-10 (Phase 7: Optimization & Integration)
- **Deep Learning Recall Model**: Integrated `YoutubeDNN` (50 epochs, trained on GPU) into `RecallFusion`.
  - Serves as the primary recall channel (weight=2.0) for personalized recommendations.
  - Implemented high-performance in-memory embedding pre-computation on startup.
- **Deduplication Logic**:
  - Added strict deduplication in `RecommendationService`.
  - Filters out books already in user's favorites.
  - Filters out duplicate titles (even if different ISBNs) using an ISBN-Title mapping.

### Changed - 2026-01-10
- **Performance Optimization**: 
  - Enabled "Resource Pre-warming" in backend `startup_event`.
  - Reduced cold-start latency for `/api/recommend/personal` from ~15s (lazy load) to **~19ms** (warm).
  - Backend startup time increased slightly (`make run`) to ensure instant user response.
- **Code Refactoring**:
  - Moved `YoutubeDNNRecall` integration from placeholder to active use in `fusion.py`.
  - Optimized vector DB loading to handle `langchain` deprecation warnings.

### Fixed - 2026-01-10
- **Dependency Issues**: Resolved `ModuleNotFoundError: langchain_openai` by updating environment.
- **Critical Bugs**:
  - Fixed `TypeError` in `list_favorites` handling (dict vs list mismatch).
  - Fixed duplicate book display in frontend recommendations.
  - Resolved port 6006 conflicts during reload cycles.

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
- **Real-time Book Cover Fetching**: New `src/infra/cover_fetcher.py` module that fetches book covers dynamically from Google Books API and Open Library
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

