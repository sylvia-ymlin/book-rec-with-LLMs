# Interview Deep Dive: Book Recommender Analysis
**Framework**: Based on "LLM Application Landing" (SFT vs RAG) criteria.

---

## I. Project Classification
**Type**: **Agentic RAG** (Retrieval-Augmented Generation with Router Control).
*   *Not just "RAG"*: It includes a decision layer (`QueryRouter`) that changes strategy based on input.
*   *Not just "Search"*: It generates grounded responses using retrieved context.

---

## II. RAG Technical Depth (The "Meat")

### 1. Architecture Type
*   **Our Choice**: **Agentic RAG** (Router -> Branching Logic).
*   **Why?**:
    *   A simple "Retriever-Generator" chain failed on **Exact Intents** (ISBNs) and **Freshness** queries.
    *   We needed dynamic logic: "If specific ID, use precise tool; If vague feeling, use semantic tool."
*   *Common Interview Question*: "Why didn't you use GraphRAG?"
    *   *Answer*: "Overkill for MVP. Entities (Books) are independent atoms; we don't heavily rely on multi-hop relationships (e.g., 'Books written by the friend of the author of X'). Agentic Routing solved 80% of edge cases with 1% of the complexity."

### 2. Knowledge Base Construction (The Foundation)
*   **Strategy**: **Atomic Documents** (Structure-Aware).
    *   *Implementation*: Instead of fixed-size chunking (e.g., 512 tokens), we treated each **Book** as a single atomic unit.
    *   *Content Construction*: `Title` + `Author` + `Description` + `Review Highlights` + `Emotions`.
*   **Why not Fixed Chunking?**:
    *   Users search for *whole books*, not *fragments of a paragraph* inside a book description.
    *   *Trade-off*: We sacrifice granularity for context integrity.
    *   *Optimization*: We injected `Review Highlights` (User Opinions) into the text representation to allow semantic matching on "vibe" (e.g., "readers hate the ending").

### 3. Retrieval Strategy Optimization (The Core Battlefield)
*   **A. User Intent Recognition**:
    *   *Tech*: RegEx & Keyword Routing (`src/rag/router.py`).
    *   *Logic*: Distinguishes **Identificational** (ISBN), **Informational** (Topic), and **Recency** (Latest) queries.
*   **B. Hybrid Search**:
    *   *Tech*: Reciprocal Rank Fusion (RRF) of BM25 (Sparse) + Chroma (Dense).
    *   *Why*: Dense vectors are bad at exact numbers (ISBNs) and rare proper nouns. BM25 covers this blind spot.
*   **C. Reranking (Precision)**:
    *   *Tech*: Cross-Encoder (`ms-marco-MiniLM`).
    *   *Impact*: Moved semantic "noise" chunks down. Fixed the "Harry Potter Philosophy vs Sorcerer's Stone" relevance issue.
*   **D. Non-Semantic Scoring**:
    *   *Tech*: **Temporal Dynamics** (Time Decay).
    *   *Logic*: $Score \times (1 + \frac{1}{\log(Age)})$.
    *   *Why*: Relevance isn't just "Topic Match"; for technology/news, "Newness" *is* relevance.

### 4. Generation Optimization
*   **Prompt Engineering**:
    *   *Structure*: "Librarian Persona" + Strict Context Boundary ("If not in context, state general knowledge").
*   **Context Compression**:
    *   *Problem*: Multi-turn chat exhausts token windows.
    *   *Solution*: Summarization of older turns + Raw retention of recent turns.
    *   *Trade-off*: Loss of specific wording in old turns vs. ability to sustain infinite conversation.

### 5. Post-Deployment Engineering
*   **Observability**:
    *   *Tech*: Prometheus Middleware.
    *   *Metrics*: Latency (P99), Request Count, Error Rate.
*   **Feedback Loop**:
    *   *User Signal*: "Add to Favorites" serves as implicit positive feedback.
    *   *Future*: This data could train a **Reward Model** for RLHF/DPO.

---

## III. SFT Potential (Where to go next?)
*If asked: "How would you use SFT to improve this?"*

1.  **Data Design**:
    *   Construct `(User Query, Retrieved Context, Ideal Librarian Response)` triplets.
    *   **Goal**: Train the model to adopt a specific "Literary Critic" tone that default GPT-3.5 lacks.
2.  **DPO (Direct Preference Optimization)**:
    *   Use the "Refused Recommendations" (users *didn't* click) vs "Accepted Recommendations" (users added to shelf) to construct Preference Pairs ($y_w, y_l$).
    *   Fine-tune the model to align with *successful* recommendation justifications.

---

## IV. The "Golden Thread" Narrative
**Motivation**: "I wanted to solve the 'Paradox of Choice' in book discovery—users know what they feel ('sad sci-fi') but search engines only understand keywords."

**Trade-off Highlight**: "I chose an **Embedded Vector DB** (Chroma) over a Service (Pinecone) to achieve **Zero Network Latency** and simplify the Ops stack, knowing the dataset (<1M books) fits easily in memory."

**Result**: "An Agentic system that corrects its own retrieval strategy, achieving 100% recall on ISBNs while maintaining deep semantic understanding."
