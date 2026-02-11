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

## 3. System Architecture

The application is engineered as a distributed system using a microservices pattern, facilitating scalability and maintainability.

- **Inference Service (FastAPI)**: A high-performance Python web framework handling HTTP requests. It acts as the orchestration layer, managing model inference and database queries.
- **Vector Database (ChromaDB)**: A dedicated vector store for similarity search. It utilizes Hierarchical Navigable Small World (HNSW) graphs for approximate nearest neighbor search, ensuring $O(\log N)$ retrieval complexity.
- **User Interface (Gradio)**: A decoupled frontend service that consumes the REST API.
- **Containerization (Docker)**: The entire stack is containerized, ensuring environment consistency across development and production.

## 4. Experimental Results

The system was evaluated on a curated subset of the dataset.

- **Data Retention**: 95.7% of the original dataset was retained after cleaning.
- **Classification Accuracy**: The Zero-Shot classifier achieved 77.8% accuracy on a binary Fiction/Non-Fiction split.
- **Inference Latency**: The average retrieval time for a top-k semantic search ($k=50$) is <200ms on standard hardware (excluding model loading time).
- **Throughput**: Batch processing of emotion analysis achieved a rate of 8.39 books/second.

## 5. Usage and Installation

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

## 6. References

1. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
2. **RoBERTa**: Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
3. **BART**: Lewis, M., et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.
