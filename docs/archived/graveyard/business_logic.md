# Business Logic: The "Knowledge Brain" Book Assistant
> **Goal**: Transform the system from a passive book catalog into an active, intelligent "Book Concierge" capable of deep Q&A and reasoning-based discovery.

---

## 1. Core Value Proposition (The "Why")

Unlike traditional bookstores that just list metadata (Title, Author, Price), our **Knowledge Brain**:
1.  **Understands Context**: Deciphers complex user intents (e.g., "books for a heartbreak") rather than just keywords.
2.  **Answers Questions**: Acts as a knowledgeable librarian who has read every book and can answer specific questions about plot, content warnings, or reading difficulty.
3.  **Synthesizes Information**: Combines internal inventory data with external LLM knowledge to provide comprehensive advice.

---

## 2. User Journey (The "What")

### Scenario A: The "Curious Reader" (Deep Q&A)
*   **User Action**: Clicks on a specific book cover (e.g., *Thinking, Fast and Slow*) and opens the "Chat with Book" modal.
*   **User Query**: "Is this book too academic for a casual reader?"
*   **System Logic**:
    1.  **Analyze**: System understands the intent is about "Readability/Difficulty".
    2.  **Retrieve**: Fetches book description, tags, and sample reviews from vector store.
    3.  **Synthesize**: LLM combines retrieved info + generic knowledge about Daniel Kahneman.
    4.  **Respond**: "It is based on academic research but written for a general audience. However, it is dense and requires focus. If you want something lighter, I can suggest..."

### Scenario B: The "Vague Seeker" (Reasoning Search)
*   **User Action**: Types in global search bar: "I want a sci-fi that isn't focused on space battles, more about sociology."
*   **System Logic**:
    1.  **Reasoning**: LLM translates user query into semantic concepts: `Sociological Sci-Fi`, `Dystopian`, `Utopian`, `No Military Sci-Fi`.
    2.  **Search**: Performs weighted vector search on these concepts.
    3.  **Explain**: Returns results with a customized reason: *"Recommended 'The Dispossessed' because it focuses on anarcho-syndicalist society structure rather than space war."*

---

## 3. Product Scope & Constraints

| Feature | In Scope (MVP) | Out of Scope (Later) |
| :--- | :--- | :--- |
| **Interaction Model** | "Chat with Book" (Single Book Context) | Global Chatbot (Full inventory context) |
| **Knowledge Source** | Hybrid (Vector DB + LLM Internal Knowledge) | Live Internet Search (Google Search) |
| **Personalization** | Use existing favorites to tune tone | Full learning of user chat history |
| **Output** | Text answers | Voice interaction |

---

## 4. Success Metrics

1.  **Response Relevance**: Does the answer actually address the specific question?
2.  **Latency**: Chat response should start streaming within <3 seconds.
3.  **Hallucination Rate**: The bot should not invent plot points that don't exist.
