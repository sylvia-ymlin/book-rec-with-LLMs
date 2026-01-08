# Tags and Emotion Scoring

This document describes the tag generation and emotion scoring features added to enrich book metadata.

## Overview

- **Tags**: Keyword extraction from book descriptions using TF-IDF (5-8 terms per book)
- **Emotion Scores**: Five emotion dimensions (joy, sadness, fear, anger, surprise) computed via transformer model

## Data Generation

### 1. Tag Generation

Extracts thematic keywords from aggregated review text.

**Script**: `scripts/generate_tags.py`

**Usage**:
```bash
python scripts/generate_tags.py \
  --input data/books_processed.csv \
  --output data/books_processed.csv \
  --top-n 8
```

**Algorithm**:
- TF-IDF vectorization (unigrams + bigrams)
- English stopwords + domain stoplist (e.g., "book", "author", "story")
- Top-N weighted terms per book
- Semicolon-joined storage in `tags` column

**Parameters**:
- `--top-n`: Max tags per book (default: 8)
- `--max-features`: TF-IDF vocabulary size (default: 60,000)
- `--min-df`: Minimum document frequency (default: 5)
- `--max-df`: Maximum document frequency ratio (default: 0.5)

### 2. Emotion Scoring

Computes emotion intensity scores from book descriptions.

**Script**: `scripts/generate_emotions.py`

**Model**: `j-hartmann/emotion-english-distilroberta-base`

**Usage**:
```bash
# CPU
python scripts/generate_emotions.py \
  --input data/books_processed.csv \
  --output data/books_processed.csv \
  --batch-size 16

# Apple GPU (MPS)
python scripts/generate_emotions.py \
  --input data/books_processed.csv \
  --output data/books_processed.csv \
  --batch-size 8 \
  --device mps \
  --checkpoint 2000 \
  --resume
```

**Parameters**:
- `--batch-size`: Inference batch size (default: 16)
- `--device`: `mps` (Apple GPU), CUDA device id, or CPU (default)
- `--checkpoint`: Rows between checkpoint writes (default: 5000)
- `--resume`: Skip rows already scored (useful for resuming long runs)
- `--max-rows`: Limit processing to N rows (for testing)

**Output Columns**:
- `joy`: 0.0–1.0
- `sadness`: 0.0–1.0
- `fear`: 0.0–1.0
- `anger`: 0.0–1.0
- `surprise`: 0.0–1.0

**Performance**:
- ~1.1 it/s on Apple M-series GPU
- ~7 hours for 222k books (batch_size=8, MPS)
- One-time processing; results persist in CSV

## Data Schema

Updated `books_processed.csv` columns:

| Column | Type | Description |
|--------|------|-------------|
| `tags` | str | Semicolon-separated keywords (e.g., "irish;travel;humor") |
| `joy` | float | Joy emotion score (0.0–1.0) |
| `sadness` | float | Sadness emotion score (0.0–1.0) |
| `fear` | float | Fear emotion score (0.0–1.0) |
| `anger` | float | Anger emotion score (0.0–1.0) |
| `surprise` | float | Surprise emotion score (0.0–1.0) |

## API Integration

### Backend Changes

**File**: `src/recommender.py`

Added to `_format_results()`:
```python
# Parse tags
tags_raw = str(row.get("tags", "")).strip()
tags = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []

# Extract emotions
emotions = {
    "joy": float(row.get("joy", 0.0)),
    "sadness": float(row.get("sadness", 0.0)),
    "fear": float(row.get("fear", 0.0)),
    "anger": float(row.get("anger", 0.0)),
    "surprise": float(row.get("surprise", 0.0)),
}
```

**File**: `src/main.py`

Updated Pydantic model:
```python
class BookResponse(BaseModel):
    isbn: str
    title: str
    authors: str
    description: str
    thumbnail: str
    caption: str
    tags: List[str] = []
    emotions: Dict[str, float] = {}
```

### API Response Example

```json
{
  "recommendations": [
    {
      "isbn": "0001849883",
      "title": "Bury My Bones But Keep My Words",
      "authors": "Deborah Savage, Tony Fairman",
      "tags": ["paulsen", "otters", "searches", "gary", "brian"],
      "emotions": {
        "joy": 0.020,
        "sadness": 0.004,
        "fear": 0.012,
        "anger": 0.006,
        "surprise": 0.086
      }
    }
  ]
}
```

## UI Display

### Search Results Grid

Each book card displays:
- **Dominant emotion label**: Emotion with highest score (bottom-right badge)
- Example: "joy", "sadness", "fear"

**Implementation** (`web/src/App.jsx`):
```jsx
{book.emotions && Object.keys(book.emotions).length > 0 ? (
  <span className="text-[9px] bg-[#f8f9fa] border border-[#eee] px-1 text-[#999] capitalize">
    {Object.entries(book.emotions).reduce((a, b) => a[1] > b[1] ? a : b)[0]}
  </span>
) : (
  <span className="text-[9px] bg-[#f8f9fa] border border-[#eee] px-1 text-[#999]">—</span>
)}
```

### Book Detail Modal

Two new sections:

**1. Key Themes**
- Displays all extracted tags as badges
- Shows "No themes found" if tags empty

**2. Emotional Tone**
- Five horizontal bars showing emotion scores
- Bar width = score percentage (0–100%)
- Format: `emotion_name | [bar] | percentage`

**Implementation** (`web/src/App.jsx`):
```jsx
<div className="space-y-2">
  <h4>Emotional Tone</h4>
  <div className="space-y-2 p-3 bg-[#faf9f6] border border-[#eee]">
    {selectedBook.emotions && Object.entries(selectedBook.emotions).map(([emotion, score]) => (
      <div key={emotion} className="flex items-center gap-2">
        <span className="text-[9px] font-bold text-gray-500 w-16 capitalize">{emotion}</span>
        <div className="flex-grow bg-white border border-[#eee] h-2 relative overflow-hidden">
          <div 
            className="h-full bg-[#b392ac] transition-all"
            style={{ width: `${Math.round(score * 100)}%` }}
          />
        </div>
        <span className="text-[8px] text-gray-400 w-10 text-right">{Math.round(score * 100)}%</span>
      </div>
    ))}
  </div>
</div>
```

## Future Improvements

- **Incremental updates**: Score only new books instead of full dataset
- **Smaller model**: Try lightweight emotion classifiers (faster inference)
- **Multi-label tags**: Use text classification for predefined categories
- **Tag filtering**: Allow users to filter by specific tags in search
- **Emotion-based sorting**: Sort results by dominant emotion match
- **Caching**: Cache emotion inference results in Redis for API speedup

## Dependencies

```
scikit-learn  # TF-IDF vectorization
transformers  # Emotion classification
torch         # Model inference
tqdm          # Progress bars
```

## Notes

- Tags and emotions are **one-time computed** and stored in CSV
- No re-computation on API requests (instant serving)
- CSV file (242MB) is in `.gitignore` (too large for GitHub)
- To regenerate on a new machine, run both scripts sequentially:
  1. `generate_tags.py` (~5 minutes)
  2. `generate_emotions.py` (~7 hours on MPS for full dataset)
