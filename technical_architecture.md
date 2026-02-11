# Technical Architecture: The "Knowledge Brain" Implementation

> **Goal**: Bridge the gap between the chaotic legacy code and the "AI Book Concierge" vision by implementing a modular, RAG-friendly backend.

---

## 1. System Overview

We are upgrading from a **Metadata Search Engine** to a **Generative AI Platform**.

### High-Level Stack
- **Frontend**: React (Vite) [Existing] -> Needs new Chat UI components.
- **Backend**: FastAPI [Existing] -> Needs new `AgentRouter` and Streaming Response support.
- **LLM Layer**: **LangChain** (New) -> Orchestrates the reasoning.
- **Memory**: **ChromaDB** [Existing] -> Serves as the "Retriever".

---

## 2. Technical Roadmap

### Milestone 1: Environment Stabilization (The Foundation)
*   **Objective**: Get `src/main.py` running locally without crashing.
*   **Action**: 
    - Create clean `conda` env with pinned `torch`, `sentence-transformers`, `fastapi`, and `langchain`.
    - Fix M1 Mutex Lock: Pin `tokenizers` and `transformers` versions, exclude `tensorflow`.

### Milestone 2: The "Real" Agent (The Upgrade)
*   **Objective**: Replace `src/agent/` (detached code) with a working LangChain module.
*   **Architecture**:
    - **Interface**: `LLMService` abstract class.
    - **Impl**: `OpenAILLM` (easy start), `OllamaLLM` (local option).
    - **Chain**: `RetrievalQA` chain that takes `{book_context}` + `{user_question}` -> `{answer}`.
*   **New Endpoint**: `POST /chat/completions` (Streaming output).

### Milestone 3: Connecting the Dots (The Integration)
*   **Objective**: Wire the Agent to the API and Frontend.
*   **Backend**: Update `src/main.py` to route chat requests to `src/agent/service.py`.
*   **Frontend**: Update `DetailModal.jsx` to call the new chat endpoint instead of just showing static text.

---

## 3. Component Design

### 3.1 Directory Restructuring (Proposed)
We need to clean up the `src` folder to reflect this logic.

```text
src/
├── api/                  # FastAPI routers
│   ├── routes.py         # Main endpoints
│   └── dependencies.py   # Auth/DB injection
├── core/
│   ├── config.py         # Settings
│   └── llm.py            # ✨ NEW: LLM Factory (OpenAI/Ollama)
├── services/
│   ├── catalog.py        # Logic for Search/Recs (Old recommender.py)
│   ├── agent.py          # ✨ NEW: RAG/Chat logic
│   └── marketing.py      # Logic for Persona/Highlights
└── data/                 # Data access layer
    └── vector_db.py      # Chroma wrapper
```

### 3.2 Data Flow: Advanced RAG Request (BYOK Implementation)
1.  **Auth (Frontend)**: User inputs API Key in Settings Modal -> Stored in LocalStorage/Session.
2.  **Request**: `POST /chat` with header `X-LLM-Key: sk-...`
3.  **Pre-processing (Self-Querying)**: LLM analyzes query to extract filters (e.g., "sci-fi" -> `category: Science Fiction`).
4.  **Retrieval (Hybrid)**: 
    - Semantic Search (ChromaDB)
    - Metadata Filtering (based on step 3)
5.  **Re-ranking**: `CrossEncoder` scores the retrieved documents against the query to sharpen precision.
6.  **Generation**: Top-K re-ranked chunks + System Prompt (Persona) -> LLM Answer.
7.  **Response**: Stream back to user.

### 3.3 Directory Restructuring (Proposed)
```text
src/
├── api/
├── core/
│   ├── llm.py            # LLM Factory
│   └── reranker.py       # ✨ NEW: Cross-Encoder Wrapper
├── services/
│   ├── query_engine.py   # ✨ NEW: Self-Querying Logic
│   └── agent.py
...
```


---

## 4. Immediate Next Steps (The "To-Do")

1.  **Fix Environment**: Create `environment.yml` to lock dependencies.
2.  **Verify Data**: Ensure `books_with_emotions.csv` loads correctly.
3.  **Implement `LLMService`**: Write the basic adapter for OpenAI/Mock to test without heavy GPU.
