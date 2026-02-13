# Review Highlights Feature

## Overview

Added semantic sentence extraction to display representative reader reviews for each book. This feature enhances book discovery by showcasing authentic reader voices.

## Implementation

### 1. Data Generation (Server-side)

**Script**: `scripts/extract_review_sentences.py`

**Process**:
- Splits book descriptions into sentences using regex
- Uses `sentence-transformers/all-MiniLM-L6-v2` for sentence embeddings
- Clusters similar sentences via cosine similarity (threshold: 0.8)
- Extracts representative sentences from each cluster (top 5 per book)
- Stores as semicolon-separated `review_highlights` column in CSV

**Execution**:
```bash
# Run in container with GPU
export HF_ENDPOINT=https://hf-mirror.com
python scripts/extract_review_sentences.py \
  --input data/books_processed.csv \
  --output data/books_processed.csv \
  --top-n 5 \
  --similarity-threshold 0.8 \
  --device 0 \
  --batch-size 128
```

**Performance**: ~17 minutes for 222k books on GPU (211 it/s)

### 2. Backend Integration

**Files Modified**:
- `src/recommender.py`: Parse `review_highlights` from CSV, split by semicolon
- `src/app/main.py`: Add `review_highlights: List[str]` to `BookResponse` model

**Code**:
```python
# Parse review highlights from semicolon-separated string
highlights_raw = str(row.get("review_highlights", "")).strip()
review_highlights = [h.strip() for h in highlights_raw.split(";") if h.strip()]
```

### 3. Frontend Display

**File**: `web/src/App.jsx`

**Location**: Left column, bottom section (below Rating/Mood)

**Features**:
- Displays up to 3 representative sentences
- Bullet-point format with `-` prefix
- Complete sentences: `- "[sentence]"`
- Incomplete sentences: `- "...[sentence]"` (auto-detected via regex `/^[A-Z]/`)
- Styling: 10px italic gray text

**Layout**:
```jsx
{selectedBook.review_highlights && selectedBook.review_highlights.length > 0 && (
  <div className="w-full mt-auto space-y-2 text-left">
    {selectedBook.review_highlights.slice(0, 3).map((highlight, idx) => {
      const isCompleteSentence = /^[A-Z]/.test(highlight.trim());
      const prefix = isCompleteSentence ? '' : '...';
      return (
        <p key={idx} className="text-[10px] text-[#666] leading-relaxed italic pl-2">
          - "{prefix}{highlight}"
        </p>
      );
    })}
  </div>
)}
```

## Related Changes

### Rating Display Enhancement

**Problem**: Hardcoded rating value of 4 stars for all books

**Solution**:
- Added `average_rating` field to backend API response
- Display format: `4.3` (1 decimal) + filled stars
- Moved rating display into AI highlight box (pink desc_block)

**Frontend mapping**:
```javascript
rating: r.average_rating || 0,  // Keep float, no rounding
```

**Display**:
```jsx
<span>{selectedBook.rating ? selectedBook.rating.toFixed(1) : '0.0'}</span>
<div className="flex gap-0.5 text-[#f4acb7]">
  {[1,2,3,4,5].map(i => <Star key={i} className={`w-3 h-3 ${i <= selectedBook.rating ? 'fill-current' : ''}`} />)}
</div>
```

### Layout Adjustments

- Grid ratio: 4:8 → 5:7 (more space for left column)
- Rating/Mood: Changed from vertical stack to consolidated display
- Rating moved into desc_block (AI highlight box)
- Review highlights positioned at bottom with `mt-auto`

## Data Schema

**CSV Column**: `review_highlights` (string, semicolon-separated)

**Example**:
```
"Having been brought up on the notion...;It transpires, some years ago...;This is a work full of wisdom..."
```

**API Response**:
```json
{
  "review_highlights": [
    "Having been brought up on the notion that Elizabeth Barrett Browning was the slighter poet...",
    "It transpires, some years ago, Clarke hosted two hugely successful British television series...",
    "This is a work full of wisdom and unusual perspectives."
  ],
  "average_rating": 3.716216
}
```

## Notes

- Review highlights are pre-computed and stored in CSV (no runtime extraction)
- Data file `books_processed.csv` (~243MB) must be regenerated after container rebuild
- Use `scp` to transfer processed CSV back to local machine
- HuggingFace mirror (`HF_ENDPOINT`) required for model download in restricted networks

## Future Improvements

- Cache sentence embeddings to speed up re-generation
- Add sentiment analysis to highlights (positive/critical)
- Filter highlights by relevance to user query
- Display highlight source (verified purchase vs. regular review)
