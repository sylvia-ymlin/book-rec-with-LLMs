# Project Analysis: Semantic Book Recommender with LLM Embeddings

A structured analysis of the Book Recommendation System using a three-layer framework: Surface, Middle, and Deep.

---

## 1. Surface Level

**Core Question: What is visible? What are the inputs and outputs?**

- **Project Type:** End-to-End AI E-Commerce / Discovery Platform.
- **User Interface:**  
  - **Frontend:** Modern SPA built with React 18, Vite, and Tailwind CSS.
  - **Aesthetics:** "Paper Shelf" (纸间留白) design philosophy - minimalist, literary look.
  - **Interactions:** Semantic search bar, infinite grid of book covers, "Add to Favorites" toggle, and a "Private Library" view.
- **Key Features:**
  - **Semantic Search:** Users query natural language (e.g., "sad story about winter") instead of keywords.
  - **Personalize:** "My Library" tracks favorites and generates a "User Persona" (e.g., "You love sci-fi").
  - **Highlights:** Generates "Personalized Selling Points" explaining *why* a book matches your taste.
- **Inputs:** User query strings, click events (Add to Favorites).
- **Outputs:** Ranked list of books with covers, generated captions, and personalized highlights.

---

## 2. Middle Level

**Core Question: How are the components orchestrated? What is the logic flow?**

### 2.1 Logic Flow (Discovery -> Personalization)
1.  **Search**: User input -> `GET /recommend` -> **FastAPI**.
2.  **Retrieval**: FastAPI -> **VectorDB (Chroma)** -> Performs similarity search using `all-MiniLM-L6-v2` embeddings.
3.  **Filtration**: Results -> Filtered by Category/Tone -> Top N Candidates.
4.  **Enrichment**: Candidates -> **Highlights Engine** -> Checks User Persona -> dynamic caption generation.
5.  **Response**: JSON payload with metadata + image URLs -> React Frontend.

### 2.2 Component Interaction
- **Frontend (Web)**: React interacts with FastAPI via REST. Handles CORS and state management (favorites).
- **Backend (API)**: FastAPI orchestrates the flow.
  - `recommender.py`: The "Brain". Coordinates Search + Persona.
  - `vector_db.py`: The "Memory". Wraps ChromaDB and HuggingFace Embeddings.
  - `marketing/`: The "Creative". Generates personas and highlights.
- **Data Persistence**: 
  - **ChromaDB**: Persists vector embeddings (read-only in serving).
  - **JSON**: `user_profiles.json` stores user favorites (Mock Database).

---

## 3. Deep Level

**Core Question: Technical architecture, dependencies, and potential bottlenecks.**

### 3.1 Technical Stack Analysis
- **Language**: Python 3.10+ (Backend), JavaScript/ES6 (Frontend).
- **LLM/NLP**: 
  - `sentence-transformers`: For generating embeddings (Local CPU execution).
  - `langchain`: Used for VectorStore abstraction.
  - **Note**: Currently utilizes *Rule-Based* logic for "Personalized Highlights," not active LLM generation at runtime (to save latency/cost).
- **Database**:
  - **ChromaDB**: Embedded vector store. Good for <100k docs.
  - **File System**: Used for User Profiles and Images (no SQL DB used yet).
- **Performance**:
  - **Caching**: Redis is mentioned in docs but optional in code (`src/infra/config.py` has fallback).
  - **Indexer**: `data/scripts/init_db.py` handles the heavy lifting of ETL (2.8GB CSV -> Vectors).

### 3.2 Key Complexity & Pain Points
1.  **Environment Dependency Hell**: 
    - Conflicts between `tensorflow` (transitive) and `torch` on M1/M2 Macs (Mutex Deadlocks).
    - Solution: Pin dependencies and strip unused libs (as seen in `interview_prep.md`).
2.  **Data Volume**:
    - Raw dataset is ~3GB. Loading it into Pandas requires significant RAM.
    - Solution: The project pre-processes this into lighter CSVs (`books_basic_info.csv`).
3.  **False "AI"**: 
    - The "Generative Marketing" is currently rule-based statistics (counting authors).
    - **Optimization Opportunity**: Integrate a real LLM (e.g., Ollama/OpenAI) for proper RAG-based highlights.

### 3.3 Optimization Roadmap
1.  **True RAG Integration**: Replace regex/rule-based highlights with an LLM that reads the book description and user persona to write a custom pitch.
2.  **Database Migration**: Move `user_profiles.json` to SQLite/PostgreSQL for multi-user support.
3.  **Hardware Acceleration**: Ensure `mps` (Metal Performance Shaders) is used on Mac for Vector Search/Embedding to reduce latency.

---
