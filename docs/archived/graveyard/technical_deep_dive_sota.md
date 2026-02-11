# Technical Deep Dive: SOTA Techniques for Advanced RAG & SFT
**Date**: 2026-01-08
**Motivation**: Address the remaining gaps in the Book Recommender to achieve "Resume-Grade" technical depth.

---

## Part I: SFT Data Pipeline (Style Alignment)

### 1.1 Problem Definition
**Current State**: The LLM responds in a generic, corporate tone.
**Desired State**: The LLM should speak like a passionate *Literary Critic* — emotional, opinionated, evocative.

**Why SFT (not just Prompting)?**
- Prompting can only do so much ("Be enthusiastic") — it doesn't teach the model *how* critics structure their arguments.
- SFT embeds the *style distribution* directly into the model's weights.

### 1.2 SOTA Technique: Self-Instruct with LLM-as-a-Judge

**References**:
- [Self-Instruct (Wang et al., 2022)](https://arxiv.org/abs/2212.10560): Generate instructions from seed data.
- [UltraChat (Ding et al., 2023)](https://arxiv.org/abs/2305.14233): Large-scale multi-turn dialogue synthesis.
- [Alpaca (Stanford, 2023)](https://crfm.stanford.edu/2023/03/13/alpaca.html): Instruction-following via distillation.

**Pipeline Design**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SFT Data Synthesis Pipeline                 │
├─────────────────────────────────────────────────────────────────┤
│  1. Seed Selection                                              │
│     - Sample 1000 high-emotion reviews (rating=5, length>200)   │
│     - Filter for reviews with subjective language (e.g. "I felt")│
├─────────────────────────────────────────────────────────────────┤
│  2. Instruction Evolution (Self-Instruct)                       │
│     - Prompt GPT-4: "Given this review, generate a user question│
│       that would have prompted this recommendation."            │
│     - Result: (Query, Review) pairs                             │
├─────────────────────────────────────────────────────────────────┤
│  3. Response Transformation                                     │
│     - Prompt GPT-4: "Rewrite the review as if you are an AI     │
│       book concierge, keeping the emotional depth and specific  │
│       evidence. Do NOT add external knowledge."                 │
│     - Result: (Query, AI Response) pairs                        │
├─────────────────────────────────────────────────────────────────┤
│  4. Quality Filtering (LLM-as-a-Judge)                          │
│     - Prompt GPT-4: "Rate this dialogue on: Empathy (1-10),     │
│       Specificity (1-10), Critique Depth (1-10). Explain."      │
│     - Threshold: Keep only samples with average >= 8.           │
├─────────────────────────────────────────────────────────────────┤
│  5. DPO Pair Construction (Optional)                            │
│     - For each (Query, Response), generate a "Rejected" response│
│       by prompting GPT-4: "Rewrite this in a boring, generic way"│
│     - Result: (Query, Chosen, Rejected) triplets for DPO.       │
└─────────────────────────────────────────────────────────────────┘
```

**Expected Output**:
- `data/sft/literary_critic_train.jsonl`: ~800 high-quality (Query, Response) pairs.
- `data/dpo/preference_pairs.jsonl`: ~500 (Chosen, Rejected) pairs.

**Interview Talking Point**:
> "I didn't just use the dataset as-is. I designed a data synthesis pipeline to evolve raw user reviews into instruction-following format, then applied LLM-as-a-Judge to filter for quality. This is the same approach used in Stanford Alpaca and Meta's Llama-2 post-training."

---

## Part II: Advanced RAG (Small-to-Big Retrieval)

### 2.1 Problem Definition
**Current State**: Each book is indexed as ONE atomic chunk (~500 tokens).
**Failure Case**: User asks "Book where the narrator is unreliable and you only realize at the end" — this detail is buried in a *specific review*, not the book description.

**Why Small-to-Big?**
- Small chunks have higher semantic precision (they match the query better).
- But small chunks alone lack *context* — the LLM needs the full book info to answer.
- Solution: **Retrieve Small, Return Big**.

### 2.2 SOTA Technique: Parent-Child Document Retrieval

**References**:
- [LlamaIndex: Recursive Retrieval](https://docs.llamaindex.ai/): Parent-child document linking.
- [RAPTOR (Sarthi et al., 2024)](https://arxiv.org/abs/2401.18059): Hierarchical tree-based indexing.
- [Multi-Vector Retriever (LangChain)](https://python.langchain.com/): Separate index for summaries vs full docs.

**Architecture Design**:

```
┌─────────────────────────────────────────────────────────────────┐
│                  Small-to-Big Retrieval Architecture            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               CHILD INDEX (Review Chunks)               │    │
│  │  - Each review split into 1-3 sentences (~100 tokens)   │    │
│  │  - Metadata: { "parent_isbn": "9780123456789" }         │    │
│  │  - Stored in: ChromaDB (collection: "review_chunks")    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                    │
│                            │ similarity_search(query)           │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               MATCH: Review Chunk #42                   │    │
│  │  "The twist about the simulation was mind-blowing..."   │    │
│  │  Metadata: { "parent_isbn": "9780123456789" }           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                    │
│                            │ lookup parent_isbn                 │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               PARENT INDEX (Full Books)                 │    │
│  │  - Full book metadata: Title, Author, Description,      │    │
│  │    Review Highlights, Categories, Emotions              │    │
│  │  - Stored in: ChromaDB (collection: "books")            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               RETURN: Full Book Context                 │    │
│  │  Title: "Dark Matter"                                   │    │
│  │  Author: "Blake Crouch"                                 │    │
│  │  Description: "A physicist is abducted into..."         │    │
│  │  (Sent to LLM as RAG context)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Plan**:

1. **Chunking Script** (`scripts/chunk_reviews.py`):
   - Read `review_highlights.txt` (format: `ISBN review_text`).
   - Split each review into sentences using NLTK or spaCy.
   - Output: `data/review_chunks.jsonl` with `{ "text": "...", "isbn": "..." }`.

2. **Dual Index Initialization** (`scripts/init_dual_index.py`):
   - Create ChromaDB collection `review_chunks` with the sentence-level data.
   - Keep existing `books` collection for parent lookup.

3. **Retrieval Logic Update** (`src/vector_db.py`):
   - New method: `small_to_big_search(query, k=5)`.
   - Step 1: Query `review_chunks` collection → Get top-k chunk matches.
   - Step 2: Extract unique `parent_isbn` from matches.
   - Step 3: Fetch full book info from `books` collection using ISBN filter.

**Interview Talking Point**:
> "I implemented a hierarchical retrieval system inspired by LlamaIndex's Parent-Child pattern. Instead of indexing entire books, I indexed individual review sentences for high-precision matching, then recursively retrieved the parent book context. This solved the 'needle in a haystack' problem for detail-oriented queries."

---

## Part III: Query Expansion (HyDE - Future)

### 3.1 Problem Definition
**Failure Case**: User asks "That blue robot book" but the book description says "android with azure plating".

### 3.2 SOTA Technique: Hypothetical Document Embeddings (HyDE)

**Reference**: [HyDE (Gao et al., 2022)](https://arxiv.org/abs/2212.10496)

**Concept**: Before searching, generate a *hypothetical* document that would answer the query, then embed *that* instead of the query.

**Future Implementation**:
```python
def hyde_search(query: str) -> List[Document]:
    # Step 1: Generate hypothetical document
    prompt = f"Write a detailed book description that would perfectly match: {query}"
    hypothetical_doc = llm.invoke(prompt)
    
    # Step 2: Embed the hypothetical doc (not the query)
    results = vector_db.search(hypothetical_doc, k=10)
    return results
```

**Status**: Deferred to Phase 7. Current focus is Small-to-Big.

---

## Implementation Priority

| Priority | Feature | File | Status |
|----------|---------|------|--------|
| 1 | SFT Data Generator | `src/data_factory/generator.py` | TODO |
| 2 | LLM Judge | `src/data_factory/judge.py` | TODO |
| 3 | Review Chunker | `scripts/chunk_reviews.py` | TODO |
| 4 | Small-to-Big Index | `scripts/init_dual_index.py` | TODO |
| 5 | Small-to-Big Search | `src/vector_db.py` | TODO |
| 6 | HyDE | `src/core/hyde.py` | Deferred |

---

## Summary

This document establishes the **technical rationale** for two major upgrades:

1. **SFT Pipeline**: Not just "training a model" but designing a *data factory* with quality control — demonstrating Data-Centric AI thinking.

2. **Small-to-Big RAG**: Not just "adding more data" but restructuring the *retrieval topology* — demonstrating Systems Architecture thinking.

Both are aligned with 2024 SOTA practices and provide concrete talking points for MLE interviews.
