# Refactoring Guide: Large Files

**Status**: Active
**Audience**: Engineers planning refactoring work
**Last Updated**: 2026-02-12
**Priority**: P2 (Quality improvement, not blocking)

---

## Overview

Several files in the codebase exceed 300 lines. While not critical, breaking them into smaller modules improves:
- **Readability**: Easier to understand each component
- **Testability**: Smaller units = easier to test in isolation
- **Maintainability**: Changes have smaller blast radius

**Industry Standard**: Files <300 lines, functions <50 lines.

---

## Files Requiring Refactoring

| File | Lines | Complexity | Priority | Suggested Split |
|------|-------|------------|----------|-----------------|
| `src/ranking/features.py` | 470 | High | P0 | Extract feature groups into separate modules |
| `src/core/web_search.py` | 427 | Medium | P1 | Separate API client from result parsing |
| `src/vector_db.py` | 368 | High | P0 | Split hybrid search from small-to-big retrieval |
| `src/recall/sasrec_recall.py` | 337 | Medium | P2 | Extract embedding logic |
| `src/recall/embedding.py` | 333 | Medium | P2 | Extract training vs inference |
| `src/services/recommend_service.py` | 313 | High | P1 | Extract diversity reranking orchestration |
| `src/core/recommendation_orchestrator.py` | 293 | Medium | P2 | Extract query preprocessing |
| `src/core/metadata_store.py` | 288 | Low | P3 | Already well-structured (mostly SQL) |

---

## Refactoring Strategy: features.py (Example)

### Current Structure (470 lines)

```python
class FeatureEngineer:
    def __init__():
        # 20 lines

    def _load_sasrec_features():
        # 35 lines

    def _load_user_sequences():
        # 20 lines

    def load_base_data():
        # 30 lines

    def generate_features():
        # 180 lines (HUGE!)
        # - User stats (15 lines)
        # - Item stats (15 lines)
        # - SASRec features (40 lines)
        # - ItemCF features (30 lines)
        # - UserCF features (30 lines)
        # - Last-N similarity (50 lines)

    def extract_features():
        # 120 lines
```

### Proposed Refactoring

**New Structure**:

```
src/ranking/
├── features.py              # Orchestrator (100 lines)
├── feature_groups/
│   ├── __init__.py
│   ├── user_features.py     # User stats extraction (40 lines)
│   ├── item_features.py     # Item stats extraction (40 lines)
│   ├── sasrec_features.py   # SASRec embedding features (60 lines)
│   ├── cf_features.py       # ItemCF + UserCF features (70 lines)
│   └── sequence_features.py # Last-N similarity (60 lines)
```

**Refactored `features.py`** (100 lines):

```python
from src.ranking.feature_groups.user_features import UserFeatureExtractor
from src.ranking.feature_groups.item_features import ItemFeatureExtractor
from src.ranking.feature_groups.sasrec_features import SASRecFeatureExtractor
from src.ranking.feature_groups.cf_features import CFFeatureExtractor
from src.ranking.feature_groups.sequence_features import SequenceFeatureExtractor

class FeatureEngineer:
    """Orchestrates feature extraction from multiple sources."""

    def __init__(self, data_dir='data/rec', model_dir='data/model/recall'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

        # Initialize sub-extractors
        self.user_extractor = UserFeatureExtractor(data_dir)
        self.item_extractor = ItemFeatureExtractor(data_dir)
        self.sasrec_extractor = SASRecFeatureExtractor(data_dir, model_dir)
        self.cf_extractor = CFFeatureExtractor(data_dir, model_dir)
        self.seq_extractor = SequenceFeatureExtractor(data_dir)

    def load_base_data(self):
        """Load all feature extractors."""
        self.user_extractor.load()
        self.item_extractor.load()
        self.sasrec_extractor.load()
        self.cf_extractor.load()
        self.seq_extractor.load()

    def generate_features(self, user_id, candidate_item, **overrides):
        """Generate feature vector by delegating to sub-extractors."""
        feats = {}

        # Delegate to each extractor
        feats.update(self.user_extractor.extract(user_id))
        feats.update(self.item_extractor.extract(candidate_item))
        feats.update(self.sasrec_extractor.extract(user_id, candidate_item, **overrides))
        feats.update(self.cf_extractor.extract(user_id, candidate_item))
        feats.update(self.seq_extractor.extract(user_id, candidate_item, **overrides))

        return feats
```

**Benefits**:
- Each extractor is <70 lines (easy to understand)
- Testable in isolation: `test_sasrec_features.py`
- Clear separation of concerns
- Easy to add new feature groups

---

## Refactoring Checklist

**Before Refactoring**:
- [ ] Read existing code thoroughly
- [ ] Identify natural module boundaries (feature groups, API layers, etc.)
- [ ] Check test coverage (write tests if missing)
- [ ] Document current behavior (input/output examples)

**During Refactoring**:
- [ ] Create new modules with single responsibility
- [ ] Move code incrementally (one function at a time)
- [ ] Run tests after each move
- [ ] Update imports in dependent files
- [ ] Add docstrings to new modules

**After Refactoring**:
- [ ] Run full test suite
- [ ] Check code coverage (should not decrease)
- [ ] Update ARCHITECTURE.md if structure changed
- [ ] Benchmark performance (ensure no regression)
- [ ] Update this guide with lessons learned

---

## Refactoring Patterns

### Pattern 1: Extract Feature Groups

**When**: Large function with distinct sections (user features, item features, etc.)

**How**:
1. Identify groups: user stats, item stats, embeddings, CF scores
2. Create one class per group: `UserFeatureExtractor`, `ItemFeatureExtractor`
3. Each class has `load()` and `extract()` methods
4. Main class delegates to extractors

**Example**: `features.py` → `feature_groups/*`

---

### Pattern 2: Split API Client from Parser

**When**: File handles both HTTP requests and response parsing

**How**:
1. Create `api_client.py`: HTTP calls, retries, auth
2. Create `parsers.py`: Parse JSON responses into domain objects
3. Main file orchestrates: `client.fetch()` → `parser.parse()`

**Example**: `web_search.py` → `google_books_client.py` + `result_parser.py`

---

### Pattern 3: Separate Training from Inference

**When**: File contains both model training code and inference code

**How**:
1. Create `trainer.py`: Fit model, save weights
2. Create `predictor.py`: Load weights, predict
3. Shared utilities in `model_utils.py`

**Example**: `sasrec_recall.py` → `sasrec_trainer.py` + `sasrec_predictor.py`

---

### Pattern 4: Extract Helper Functions

**When**: Function >50 lines with reusable logic

**How**:
1. Identify helper logic (e.g., "compute similarity", "filter results")
2. Extract to `_helper_function()` or `utils.py`
3. Main function becomes high-level orchestration

**Example**:
```python
# Before (80 lines)
def recommend(self, user_id, k=50):
    # 1. Get history (15 lines)
    # 2. Query similar items (20 lines)
    # 3. Aggregate scores (25 lines)
    # 4. Filter & sort (20 lines)

# After (20 lines)
def recommend(self, user_id, k=50):
    history = self._get_user_history(user_id)
    similar_items = self._query_similar_items(history)
    scores = self._aggregate_scores(similar_items)
    return self._filter_and_sort(scores, k)
```

---

## Example Refactoring: vector_db.py

### Current (368 lines)

```python
class VectorDB:
    def __init__():
        # 20 lines

    def search():
        # 30 lines

    def hybrid_search():
        # 120 lines (complex RRF fusion logic)

    def small_to_big_search():
        # 80 lines (hierarchical retrieval)

    def _rrf_fusion():
        # 60 lines

    def _build_bm25_index():
        # 40 lines
```

### Proposed (3 files, each <150 lines)

**src/vector_db.py** (80 lines):
```python
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.hierarchical_retriever import HierarchicalRetriever

class VectorDB:
    """Facade for multiple retrieval strategies."""

    def __init__():
        self.hybrid = HybridRetriever()
        self.hierarchical = HierarchicalRetriever()

    def search(query, k=10):
        """Dense search only."""
        return self.chroma.search(query, k=k)

    def hybrid_search(query, k=10, alpha=0.5):
        """BM25 + Dense fusion."""
        return self.hybrid.search(query, k, alpha)

    def small_to_big_search(query, k=10):
        """Sentence → Book hierarchical retrieval."""
        return self.hierarchical.search(query, k)
```

**src/retrieval/hybrid_retriever.py** (150 lines):
```python
class HybridRetriever:
    """Combines BM25 (sparse) + Dense (embeddings) via RRF."""

    def __init__():
        self.bm25 = self._build_bm25_index()
        self.dense_db = Chroma(...)

    def search(query, k, alpha):
        dense_results = self._dense_search(query, k)
        sparse_results = self._sparse_search(query, k)
        return self._rrf_fusion(dense_results, sparse_results, alpha)

    def _rrf_fusion(dense, sparse, alpha):
        # 60 lines (RRF logic)

    def _build_bm25_index():
        # 40 lines
```

**src/retrieval/hierarchical_retriever.py** (130 lines):
```python
class HierarchicalRetriever:
    """Small-to-Big (sentence → book) retrieval."""

    def search(query, k):
        # 1. Find matching sentences (child nodes)
        sentences = self.sentence_db.search(query, k=50)

        # 2. Map to parent books (ISBNs)
        book_isbns = self._map_to_parents(sentences)

        # 3. Return full book context
        return self._enrich_with_metadata(book_isbns, k)
```

---

## Migration Strategy

### Incremental Refactoring (Safe)

**Week 1: Extract utilities**
- Move helper functions to `utils.py`
- No API changes
- Run tests

**Week 2: Create new modules**
- Add `feature_groups/` directory
- Copy (don't move) code to new modules
- Old code still works

**Week 3: Update imports**
- Change `features.py` to use new modules
- Mark old methods as `@deprecated`
- Update tests

**Week 4: Remove old code**
- Delete deprecated methods
- Update ARCHITECTURE.md
- Celebrate! 🎉

---

## Testing Strategy

### Before Refactoring: Capture Behavior

```python
# tests/test_features_baseline.py
def test_feature_extraction_baseline():
    """Capture current behavior before refactoring."""
    fe = FeatureEngineer()
    fe.load_base_data()

    # Test on known user/item
    feats = fe.generate_features("A123", "0123456789")

    # Save as baseline
    import json
    with open("tests/baseline_features.json", "w") as f:
        json.dump(feats, f)
```

### After Refactoring: Compare Output

```python
# tests/test_features_refactored.py
def test_feature_extraction_matches_baseline():
    """Ensure refactored code produces identical output."""
    fe = FeatureEngineer()  # New refactored version
    fe.load_base_data()

    feats = fe.generate_features("A123", "0123456789")

    # Load baseline
    import json
    with open("tests/baseline_features.json") as f:
        baseline = json.load(f)

    # Compare
    for key in baseline:
        assert key in feats, f"Missing feature: {key}"
        assert abs(feats[key] - baseline[key]) < 1e-6, f"Feature {key} mismatch"
```

---

## Performance Considerations

### Refactoring Should Not Slow Things Down

**Before Refactoring**: Benchmark
```bash
python benchmarks/benchmark.py > baseline.txt
```

**After Refactoring**: Re-benchmark
```bash
python benchmarks/benchmark.py > refactored.txt
diff baseline.txt refactored.txt
```

**Acceptable Changes**:
- Latency: ±5% (small overhead from function calls OK)
- Memory: ±10% (object overhead from new classes OK)

**Red Flags**:
- Latency increase >10% → Investigate (unnecessary copies? Extra I/O?)
- Memory increase >20% → Investigate (duplicate data? Leaks?)

---

## Common Pitfalls

### Pitfall 1: Over-Refactoring

**Problem**: Creating too many tiny files (10-20 lines each)

**Solution**: Aim for 50-150 lines per file. Smaller isn't always better.

---

### Pitfall 2: Breaking API Compatibility

**Problem**: Refactoring changes function signatures, breaking existing code

**Solution**: Use deprecation warnings
```python
import warnings

def old_method(self, x):
    warnings.warn("old_method is deprecated, use new_method", DeprecationWarning)
    return self.new_method(x)
```

---

### Pitfall 3: Not Updating Tests

**Problem**: Tests still import old paths

**Solution**: Update imports incrementally
```python
# Before
from src.ranking.features import FeatureEngineer

# After
from src.ranking.features import FeatureEngineer  # Facade
from src.ranking.feature_groups.sasrec_features import SASRecFeatureExtractor  # Specific
```

---

## When NOT to Refactor

- **Deadline pressure**: Refactoring during crunch time adds risk
- **No tests**: Write tests first, then refactor
- **Unclear requirements**: Understand the code before restructuring
- **"Just because"**: Only refactor if it improves readability/testability

**Rule of Thumb**: Refactor when you're touching the code for a feature/bug anyway. Don't refactor in isolation.

---

## Success Metrics

How do you know refactoring succeeded?

- [ ] Code is easier to understand (ask a teammate)
- [ ] Tests pass (same behavior)
- [ ] Performance unchanged (benchmark)
- [ ] Test coverage increased (easier to test small modules)
- [ ] Future changes are faster (measure time to implement next feature)

---

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) — Overall system structure
- [DEVELOPMENT.md](DEVELOPMENT.md) — Development guidelines
- [ONBOARDING.md](../ONBOARDING.md) — New contributor guide

---

## Refactoring Backlog

Track refactoring work here:

| File | Priority | Assigned | Status | Target Date |
|------|----------|----------|--------|-------------|
| `ranking/features.py` | P0 | Unassigned | Not Started | TBD |
| `core/web_search.py` | P1 | Unassigned | Not Started | TBD |
| `vector_db.py` | P0 | Unassigned | Not Started | TBD |

---

**Last Updated**: 2026-02-12
**Next Review**: After v3.0 planning
