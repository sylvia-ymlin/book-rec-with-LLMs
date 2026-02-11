# Build Guide: From Zero to Production

This guide explains how to build the entire project from scratch.

---

## Quick Start (Already Built)

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate book-rec

# 2. Validate data (check what's ready)
make data-validate

# 3. Start backend
make run  # http://localhost:6006

# 4. Start frontend
cd web && npm install && npm run dev  # http://localhost:5173
```

### New Pipeline Commands

```bash
make data-pipeline   # Run full pipeline (data + models)
make data-prep       # Data processing only (no GPU training)
make data-validate   # Check data quality
make train-models    # Train ML models only
```

---

## Full Build Pipeline

### Overview

```
Raw Data (CSV)
     │
     ├── [1] Data Processing ──────────────────────────┐
     │   ├── books_data.csv → books_processed.csv      │
     │   ├── Books_rating.csv → rec/train,val,test.csv │
     │   └── Reviews → review_chunks                   │
     │                                                  │
     ├── [2] Index Building ───────────────────────────┤
     │   ├── ChromaDB (Vector Index)                   │
     │   └── BM25 (Sparse Index)                       │
     │                                                  │
     ├── [3] Model Training ───────────────────────────┤
     │   ├── ItemCF / UserCF                           │
     │   ├── YoutubeDNN (GPU)                          │
     │   ├── SASRec (GPU)                              │
     │   └── XGBoost Ranker                            │
     │                                                  │
     └── [4] Service Startup ──────────────────────────┘
         └── FastAPI + React
```

---

## Phase 1: Environment Setup

```bash
# Clone repo
git clone <repo-url>
cd book-rec-with-LLMs

# Create conda environment
conda env create -f environment.yml
conda activate book-rec

# Install frontend dependencies
cd web && npm install && cd ..
```

---

## Phase 2: Data Preparation

### 2.1 Raw Data Requirements

Place in `data/raw/`:
- `books_data.csv` - Book metadata (title, author, description, categories)
- `Books_rating.csv` - User ratings (User_id, Id, review/score, review/time, review/text)

### 2.2 Data Processing Scripts

| Order | Script | Purpose | Output |
|:---:|:---|:---|:---|
| 0 | `clean_data.py` | HTML/encoding/whitespace cleanup | books_processed.csv (cleaned) |
| 1 | `build_books_basic_info.py` | Extract basic book info | books_basic_info.csv |
| 2 | `generate_emotions.py` | Sentiment analysis (5 emotions) | +joy,sadness,fear,anger,surprise |
| 3 | `generate_tags.py` | TF-IDF keyword extraction | +tags column |
| 4 | `split_rec_data.py` | Leave-Last-Out time split | rec/train,val,test.csv |
| 5 | `build_sequences.py` | User history → sequences | rec/user_sequences.pkl |
| 6 | `chunk_reviews.py` | Reviews → sentences | review_chunks.jsonl |

### 2.3 Script Details

#### Data Cleaning (`clean_data.py`)
- **HTML**: Remove tags, decode entities (`&amp;` → `&`)
- **Encoding**: Fix mojibake (UTF-8 corruption)
- **Unicode**: NFKC normalization
- **Whitespace**: Collapse multiple spaces/newlines
- **URLs**: Remove from text

#### Data Split (`split_rec_data.py`)
- **Strategy**: Leave-Last-Out (时序划分)
- **Filter**: Users with ≥3 interactions
- **Output**: train (oldest) → val (2nd last) → test (last)

#### Sequence Building (`build_sequences.py`)
- **Format**: `Dict[user_id, List[item_id]]`
- **Padding**: 0 reserved, IDs are 1-indexed
- **Max length**: 50 items (truncated)

```bash
# Run via unified pipeline
python scripts/run_pipeline.py --stage books

# Or manually
python scripts/data/clean_data.py --backup
python scripts/data/split_rec_data.py
python scripts/data/build_sequences.py
```

---

## Phase 3: Index Building

### 3.1 Vector Database (ChromaDB)

```bash
python scripts/data/init_dual_index.py
```

**Output**: `data/chroma_db/` (222K book vectors)

### 3.2 Review Chunks Index (Small-to-Big)

```bash
python scripts/data/extract_review_sentences.py
```

**Output**: `data/chroma_chunks/` (788K sentence vectors)

---

## Phase 4: Model Training

### 4.1 Recall Models (CPU OK)

```bash
# Build ItemCF / UserCF matrices
python scripts/model/build_recall_models.py
```

**Output**: `data/model/recall/itemcf.pkl`, `usercf.pkl`

### 4.2 YoutubeDNN (GPU Recommended)

```bash
# Train two-tower model
python scripts/model/train_youtube_dnn.py
```

**Output**: `data/model/recall/youtube_dnn.pt`

**Training**: ~50 epochs, 2048 batch, ~30 min on GPU

### 4.3 SASRec (GPU Recommended)

```bash
# Train sequence model
python scripts/model/train_sasrec.py
```

**Output**: `data/model/recall/sasrec.pt`

**Training**: ~30 epochs, ~20 min on GPU

### 4.4 XGBoost Ranker

```bash
# Train ranking model
python scripts/model/train_ranker.py
```

**Output**: `data/model/ranking/xgb_ranker.pkl`

**Training**: ~5 min on CPU

---

## Phase 5: Service Startup

### Backend

```bash
make run
# or
uvicorn src.main:app --reload --port 6006
```

**Startup Log**:
```
Loading embedding model...           # ~20s
Loaded 222003 documents             # ~10s
BM25 Index built with 222005 docs   # ~12s
Engines Initialized.                # Ready!
```

### Frontend

```bash
cd web
npm run dev
```

**Access**:
- Frontend: http://localhost:5173
- API Docs: http://localhost:6006/docs

---

## Data Flow Summary

```
data/
├── raw/
│   ├── books_data.csv          # Original book metadata
│   └── Books_rating.csv        # Original ratings
├── books_basic_info.csv        # Processed book info
├── books_processed.csv         # Full processed data
├── chroma_db/                  # Vector index (222K)
├── chroma_chunks/              # Review chunks (788K)
├── rec/
│   ├── train.csv               # 1.08M training records
│   ├── val.csv                 # 168K validation
│   ├── test.csv                # 168K test
│   ├── user_sequences.pkl      # User history
│   └── item_map.pkl            # ISBN → ID mapping
├── model/
│   ├── recall/
│   │   ├── itemcf.pkl          # ItemCF matrix
│   │   ├── usercf.pkl          # UserCF matrix
│   │   ├── youtube_dnn.pt      # Two-tower model
│   │   └── sasrec.pt           # Sequence model
│   └── ranking/
│       └── xgb_ranker.pkl      # XGBoost ranker
└── user_profiles.json          # User favorites
```

---

## Training on GPU Server

If local machine is slow, use AutoDL/Cloud:

```bash
# Sync to server
rsync -avz . user@server:/path/to/project

# On server
python scripts/model/train_youtube_dnn.py
python scripts/model/train_sasrec.py

# Sync back
rsync -avz user@server:/path/to/project/data/model ./data/
```

---

## Minimal Local Run (Without Training)

If you only have raw data but no trained models:

1. **ItemCF/UserCF** will work (built on-demand)
2. **YoutubeDNN** will be skipped (graceful degradation)
3. **SASRec features** will be 0.0
4. **XGBoost** needs to be trained or use fallback

System will run with reduced accuracy but functional.

---

*Last Updated: January 2026*
