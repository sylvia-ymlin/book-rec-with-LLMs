---
license: mit
title: Semantic-Based Book Recommendation Framework
sdk: docker
app_port: 7860
---

# Semantic-Based Book Recommendation Framework using Large Language Model Embeddings

## Abstract

This project presents a scalable, semantic-based recommendation system designed to mitigate the limitations of traditional keyword-based search. Leveraging the **Vector Space Model (VSM)** and **Large Language Models (LLMs)**, the system performs semantic information retrieval on a dataset of over 5,000 literary works. The pipeline integrates **DistilRoBERTa** for emotion classification and **BART-Large-MNLI** for zero-shot genre categorization, enabling a multi-modal filtering approach. The system is deployed as a microservices architecture using **FastAPI** for inference and **ChromaDB** for persistent vector storage, ensuring sub-second query latency and production-grade reliability.

## 1. Introduction

Recommender systems are critical in navigating large-scale information spaces. Traditional collaborative filtering approaches suffer from the *cold-start problem*, while content-based systems often rely on rigid metadata matching. This framework addresses these challenges by utilizing **Semantic Search**, allowing users to query using natural language descriptions (e.g., "a coming-of-age story about grief") rather than specific keywords. By embedding book descriptions into a high-dimensional vector space, the system captures semantic nuance, resulting in higher relevance for complex user queries.

## 2. Methodology

The implementation follows a modular pipeline consisting of Data Preprocessing, Feature Engineering, and Inference.

### 2.1 Data Preprocessing
The dataset consists of 7,000+ books with metadata including titles, authors, and summaries. Data cleaning procedures included:
- **Null Value Handling**: Removal of records with missing descriptions or critical metadata.
- **Text Normalization**: Standardization of description text (unicode normalization, whitespace handling).
- **Quality Filtration**: Exclusion of records with descriptions shorter than 25 words to ensure sufficient semantic content for embedding.

### 2.2 Vector Embeddings
Semantic search is enabled by projecting textual descriptions into a shared vector space. We utilized the `sentence-transformers/all-MiniLM-L6-v2` model, which maps sentences to a 384-dimensional dense vector space. This model was selected for its optimal balance between inference speed and semantic accuracy (performance on the 1B Sentence Embeddings Benchmark).

### 2.3 Emotion Classification
To support mood-based filtering, we implemented a transferable multi-label classification task. We utilized **DistilRoBERTa-base**, fine-tuned on the GoEmotions dataset. For each book description, the model predicts a probability distribution across 7 emotional dimensions: *Joy, Sadness, Anger, Fear, Surprise, Love, and Neutral*.

### 2.4 Zero-Shot Classification
Genre classification was automated using a **Zero-Shot Learning** approach. We employed `facebook/bart-large-mnli`, a model trained on Multi-Genre Natural Language Inference (MNLI). This allows the system to classify books into arbitrary categories (e.g., "Fiction", "History", "Science") without requiring a labeled training set for those specific classes.

# End-to-End AI E-Commerce Platform

## Abstract

This project presents a comprehensive, multi-modal recommendation and e-commerce agent platform. It integrates large-scale semantic retrieval, retrieval-augmented generation (RAG), and content safety guardrails into a unified architecture. The system demonstrates the practical application of Large Language Models (LLMs) in modern recommender systems and user interaction agents.

## Key Features

### 1. Large-Scale Semantic Recommendations
*   **Vector Retrieval**: Utilizes ChromaDB for sub-second semantic search over a catalog of 200,000+ books.
*   **Caching Infrastructure**: Implements Redis caching to optimize latency for high-frequency queries.
*   **Zero-Shot Re-ranking**: (In Progress) Evaluates candidate generation using LLM-based zero-shot reasoning.

### 2. Conversational Shopping Assistant
*   **RAG Architecture**: Retrieves relevant product context to ground LLM responses, reducing hallucinations.
*   **Intent Recognition**: Classifies user queries (e.g., search, details, comparison) to route requests effectively.

### 3. Marketing Content Generation
*   **Automated Copywriting**: Generates marketing descriptions based on product features and target audience profiles.
*   **Safety Guardrails**: Enforces content safety policies to ensure generated text adheres to brand guidelines.

## System Architecture

The project follows a microservices-inspired architecture:

*   **Frontend**: Built with Gradio 6.0, providing a multi-tab interface for distinct module interactions.
*   **Backend API**: FastAPI service orchestration (integrated within the Gradio app for demonstration).
*   **Data Layer**:
    *   **Amazon Books Dataset**: 200,000+ records processed via custom ETL pipelines.
    *   **Vector Store**: ChromaDB for embedding storage and similarity search.
    *   **Cache**: Redis for transient data storage.

## Installation and Usage

### Prerequisites
*   Python 3.10+
*   Docker and Docker Compose

### Deployment

1.  **Clone the repository**:
    ```bash
    git clone [repository-url]
    cd book-recommender
    ```

2.  **Start Services**:
    ```bash
    docker-compose up --build
    ```

3.  **Access the Interface**:
    Navigate to `http://localhost:7860` in a web browser.

## Project Structure

```text
src/
├── recommender.py   # Core recommendation logic and retrieval
├── cache.py         # Redis caching implementation
├── etl.py           # Data extraction, transformation, and loading pipeline
├── vector_db.py     # Vector database wrapper and indexing logic
├── agent/           # Conversational shopping agent module
├── marketing/       # Marketing content generation module
└── zero_shot/       # Zero-shot re-ranking experimental module
```

## Performance Benchmarks

Latency tests were conducted on the Hugging Face Spaces environment (CPU tier):
*   **Average Latency**: 0.3 - 0.4 seconds per recommendation request.
*   **Throughput**: Validated under sequential load testing.

See `benchmarks/results.md` for detailed methodology and data.

## 6. Usage and Installation

### Prerequisites
- Docker Engine 20.10+
- Docker Compose v2.0+

### Deployment
To deploy the system locally, execute the following commands:

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd book-recommender
   ```

2. **Configuration**
   Create a `.env` file with your Hugging Face API token:
   ```bash
   echo "HUGGINGFACEHUB_API_TOKEN=your_token_here" > .env
   ```

3. **Execution**
   Build and start the container orchestration:
   ```bash
   make docker-up
   ```

   The services will be available at:
   - **API Documentation**: `http://localhost:8000/docs`
   - **Web Interface**: `http://localhost:7860`

## 7. References

1. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
2. **RoBERTa**: Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
3. **BART**: Lewis, M., et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.
