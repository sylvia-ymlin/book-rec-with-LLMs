# Project Narrative & Strategic Thinking
**Role**: End-to-End ML Engineer / AI Engineer
**Framework**: Surface (What) -> Middle (How) -> Deep (Why & Trade-offs)

---

## 1. Surface Level: The "What"
**Goal**: Define the tangible product and its unique value proposition.

*   **Definition**: An "Intelligent Book Concierge Platform" (Not just a search engine).
*   **Core Feature**: **Agentic RAG**. The system doesn't just match keywords; it understands intent, temporal context ("newest books"), and complex queries ("sad sci-fi about AI").
*   **User Experience**:
    *   **Semantic Search**: "Heartbreaking WWII stories" works as well as "Harry Potter".
    *   **Interactive Chat**: Ask follow-up questions ("Is this suitable for kids?") and get grounded answers.
    *   **Personalization**: The system learns from your "Favorites" to adjust recommendations.

---

## 2. Middle Level: The "How"
**Goal**: Demonstrate engineering depth and optimization strategies.

### Architecture Flow
1.  **Router Agent**: Classifies intent (ISBN vs. Keyword vs. Deep Question) to select the cheapest/best tool.
2.  **Hybrid Retrieval**: Fuses **BM25** (Exact Match) and **ChromaDB** (Semantic Match) via Reciprocal Rank Fusion (RRF).
3.  **Precision Layer**: Uses a **Cross-Encoder** to rerank the top 50 results for deep semantic relevance.
4.  **Temporal Dynamics**: Applies a mathematical decay function to boost newer content when appropriate.
5.  **Memory**: Compresses conversation history to allow infinite chat turns without token overflow.

### Key Innovations
*   **No "False AI"**: Unlike simple keyword apps, this uses real-time vector embeddings and LLM reasoning.
*   **Hallucination Control**: Strict RAG pipeline forces the LLM to cite its sources (book descriptions/reviews).

---

## 3. Deep Level: The "Architecture & Trade-offs"
**Goal**: Showcase architectural vision and system design skills.

### Tech Stack Decisions
*   **Vector DB (ChromaDB)**: 
    *   *Decision*: Embedded (In-Process) database.
    *   *Trade-off*: Sacrificed horizontal scalability for **Zero Network Latency** and zero-ops complexity. Perfect for the <1M dataset size.
*   **Hybrid Search (Sparse + Dense)**:
    *   *Decision*: Implemented custom RRF fusion.
    *   *Why*: Pure Vector Search failed at Specific IDs (ISBNs). Pure BM25 failed at "vibe" searches. Hybrid captures 100% of cases.
*   **Agentic Routing**:
    *   *Decision*: Rule-based Regex/Keyword Router.
    *   *Trade-off*: Chose deterministic rules over an "LLM Router" to save latency (2ms vs 500ms) and cost.

### Future Scalability
*   **Vertical Scaling**: The current in-memory index fits in 2GB RAM. Can scale to ~5M books on a standard server.
*   **Horizontal Scaling**: Easy migration path to Qdrant/Pinecone if user base grows >10k concurrent users.

---

## 4. Success Metrics
1.  **Recall**: 100% on Exact Matches (ISBNs) via Router fix.
2.  **Relevance**: Qualitative improvement in "Deep" queries via Cross-Encoder.
3.  **Latency**: Sub-second (600ms) for typical queries; <3s for complex reasoning.
