import pandas as pd
from src.core.rag.vector_db import VectorDB

def run_benchmark():
    print("🚀 Starting Temporal Dynamics Benchmark...")
    
    db = VectorDB()
    
    # We use a query where 'newness' matters
    query = "latest advancements in technology and science"
    
    print(f"\nQuery: '{query}'")
    
    # 1. Standard Search
    print("\n--- Standard Search (No Temporal) ---")
    st_docs = db.hybrid_search(query, k=5, rerank=True, temporal=False)
    for d in st_docs:
        # Get Year
        isbn = d.metadata.get("isbn") or d.metadata.get("isbn13")
        if not isbn and "ISBN:" in d.page_content:
             isbn = d.page_content.split("ISBN:")[1].strip().split()[0]
        year = db.pub_years.get(str(isbn), "Unknown")
        score = d.metadata.get("relevance_score", 0.0)
        
        # Parse title
        title = d.page_content.split('\n')[0].replace("Title: ", "")[:40]
        print(f"[{year}] {title}... (Score: {score:.4f})")
        
    # 2. Temporal Search
    print("\n--- Temporal Search (Recent Boost) ---")
    tm_docs = db.hybrid_search(query, k=5, rerank=True, temporal=True)
    for d in tm_docs:
        isbn = d.metadata.get("isbn") or d.metadata.get("isbn13")
        if not isbn and "ISBN:" in d.page_content:
             isbn = d.page_content.split("ISBN:")[1].strip().split()[0]
        year = db.pub_years.get(str(isbn), "Unknown")
        # In temporal mode, score is boosted
        score = d.metadata.get("relevance_score", 0.0)
        
        title = d.page_content.split('\n')[0].replace("Title: ", "")[:40]
        print(f"[{year}] {title}... (Score: {score:.4f})")

if __name__ == "__main__":
    run_benchmark()
