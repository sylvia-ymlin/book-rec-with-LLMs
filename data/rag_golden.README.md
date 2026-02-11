# RAG Golden Test Set

Human-annotated Query-Book pairs for quantitative RAG evaluation.

## Format

CSV with columns: `query`, `isbn`, `relevance`, `notes`

- **query**: User search string (e.g., "Harry Potter", "0060959479", "books about AI")
- **isbn**: Expected relevant book ISBN (from your catalog)
- **relevance**: 1 = relevant (filter rows with relevance=1)
- **notes**: Optional annotation note

Multiple rows per query = multiple relevant books (Recall@K counts all).

## Usage

```bash
# Copy example and extend with your catalog ISBNs
cp data/rag_golden.example.csv data/rag_golden.csv

# Run evaluation
python scripts/model/evaluate_rag.py --golden data/rag_golden.csv
```

## Metrics

- **Accuracy@K**: Fraction of queries with at least one relevant book in top-K
- **Recall@K**: Fraction of relevant books (across all queries) found in top-K
- **MRR@K**: Mean reciprocal rank of first relevant hit

Target: 500+ pairs for production-quality evaluation.
