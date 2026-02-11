#!/usr/bin/env python3
"""
Dual Index Initialization Script
Creates a separate ChromaDB collection for review chunks (Small-to-Big architecture).

SOTA Reference: LlamaIndex Parent-Child Retrieval
"""
import json
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

CHUNK_PATH = "data/review_chunks.jsonl"
PERSIST_DIR = "data/chroma_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 5000


def load_chunks(path: str, limit: int = None):
    """Load chunks from JSONL file."""
    chunks = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line)
            doc = Document(
                page_content=data["text"],
                metadata={"parent_isbn": data["parent_isbn"]}
            )
            chunks.append(doc)
    return chunks


def init_chunk_index():
    """Initialize the chunk-level ChromaDB index."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "mps"},  # Use Metal on Mac
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print(f"Loading chunks from {CHUNK_PATH}...")
    chunks = load_chunks(CHUNK_PATH)
    print(f"Loaded {len(chunks)} chunks")
    
    # Create index in batches
    print(f"Creating ChromaDB index at {PERSIST_DIR}...")
    
    # First batch creates the collection
    db = Chroma.from_documents(
        documents=chunks[:BATCH_SIZE],
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="review_chunks"
    )
    
    # Add remaining in batches
    for i in tqdm(range(BATCH_SIZE, len(chunks), BATCH_SIZE), desc="Indexing"):
        batch = chunks[i:i+BATCH_SIZE]
        db.add_documents(batch)
    
    print(f"Index created with {len(chunks)} chunks.")
    print(f"Persisted to {PERSIST_DIR}")


if __name__ == "__main__":
    init_chunk_index()
