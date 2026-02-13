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
     │   ├── ItemCF / UserCF / Swing (CPU)             │
     │   ├── YoutubeDNN (GPU)                          │
     │   ├── SASRec (GPU)                              │
     │   └── LGBMRanker (CPU)                          │
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

### 2.2 Pipeline DAG (Execution Order)

**Recommended**: Use `make data-pipeline` or `python scripts/run_pipeline.py` — it defines the full DAG.

| Stage | Script | Purpose | Output |
|:---:|:---|:---|:---|
| 1 | `build_books_basic_info.py` | Merge raw books + ratings | books_basic_info.csv |
| 2 | *books_processed.csv* | From HuggingFace or manual merge of basic_info + review_highlights | books_processed.csv |
| 3 | `clean_data.py` | HTML/encoding/whitespace cleanup | books_processed.csv (cleaned) |
| 4 | `generate_emotions.py` | Sentiment analysis (5 emotions) | +joy,sadness,fear,anger,surprise |
| 5 | `generate_tags.py` | TF-IDF keyword extraction | +tags column |
| 6 | `chunk_reviews.py` | Reviews → sentences | review_chunks.jsonl |
| 7 | `split_rec_data.py` | Leave-Last-Out time split | rec/train,val,test.csv |
| 8 | `build_sequences.py` | User history → sequences | rec/user_sequences.pkl |

**Note**: `books_processed.csv` may be pre-downloaded from HuggingFace. If building from scratch, merge `books_basic_info.csv` with review data and run `extract_review_sentences.py` first.

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
python data/scripts/clean_data.py --backup
python data/scripts/split_rec_data.py
python data/scripts/build_sequences.py
```

**Script conventions**: Use `config.data_config` for paths; `scripts.utils.setup_script_logger()` for logging.

---

## Phase 3: Index Building

### 3.1 Vector Database (ChromaDB)

```bash
python data/scripts/init_dual_index.py
```

**Output**: `data/chroma_db/` (222K book vectors)

### 3.2 Review Chunks Index (Small-to-Big)

```bash
python data/scripts/extract_review_sentences.py
```

**Output**: `data/chroma_chunks/` (788K sentence vectors)

---

## Phase 4: Model Training

### 4.1 Recall Models (CPU OK)

```bash
# Build ItemCF / UserCF / Swing / Popularity
python scripts/model/build_recall_models.py
```

**Output**: `data/model/recall/itemcf.pkl`, `usercf.pkl`, `swing.pkl`, `popularity.pkl`

**Training Time** (Apple Silicon CPU):
| Model | Time |
|:---|:---|
| ItemCF (direction-weighted) | ~2 min |
| UserCF | ~7 sec |
| Swing (optimized) | ~35 sec |
| Popularity | <1 sec |

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

### 4.4 LGBMRanker (LambdaRank)

```bash
# Train ranking model (hard negative sampling from recall results)
python scripts/model/train_ranker.py
```

**Output**: `data/model/ranking/lgbm_ranker.txt`

**Training**: ~16 min on CPU (20K users sampled, 4× hard negatives, 17 features)

---

## Phase 5: Service Startup

### Backend

```bash
make run
# or
uvicorn src.app.main:app --reload --port 6006
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
│   │   ├── itemcf.pkl          # ItemCF matrix (direction-weighted)
│   │   ├── usercf.pkl          # UserCF matrix
│   │   ├── swing.pkl           # Swing matrix
│   │   ├── popularity.pkl      # Popularity scores
│   │   ├── youtube_dnn.pt      # Two-tower model
│   │   └── sasrec.pt           # Sequence model
│   └── ranking/
│       └── lgbm_ranker.txt     # LGBMRanker (LambdaRank)
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

1. **ItemCF/UserCF/Swing** will work (CPU-trained on-demand)
2. **YoutubeDNN** will be skipped (graceful degradation)
3. **SASRec features** will be 0.0
4. **LGBMRanker** needs to be trained or use recall-score fallback

System will run with reduced accuracy but functional.

---

*Last Updated: January 2026*
