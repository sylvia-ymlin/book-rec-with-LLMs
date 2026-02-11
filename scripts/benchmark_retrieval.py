import time
import pandas as pd
from typing import List
from src.vector_db import VectorDB

def run_benchmark():
    print("🚀 Starting Retrieval Benchmark (BASELINE)...")
    
    # Load Title Mapping
    try:
        books_df = pd.read_csv("data/books_processed.csv")
        # Ensure string ISBN for matching
        books_df['isbn'] = books_df['isbn'].astype(str)
        isbn_map = books_df.set_index('isbn')['title'].to_dict()
        print(f"📚 Loaded {len(isbn_map)} titles for mapping.")
    except Exception as e:
        print(f"⚠️ Failed to load books_processed.csv: {e}")
        isbn_map = {}

    db = VectorDB()
    
    # ... (Test Cases preserved) ...
    test_queries = [
        # 1. Semantic (Dense should win)
        {"type": "Semantic", "query": "books about finding love in unexpected places"},
        {"type": "Semantic", "query": "scary stories that keep you up at night"},
        
        # 2. Keyword/Proper Noun (Dense might struggle)
        {"type": "Keyword", "query": "Harry Potter"}, 
        {"type": "Keyword", "query": "Python Programming"}, 
        {"type": "Keyword", "query": "Jane Austen"}, 
        
        # 3. Exact Match / ISBN
        {"type": "Exact", "query": "0060959479"}, 
    ]
    
    results = []
    
    for case in test_queries:
        q = case["query"]
        print(f"\nScanning: '{q}' ({case['type']})...")
        
        start_time = time.time()
        docs = db.search(q, k=5)
        duration = (time.time() - start_time) * 1000
        
        # Capture simplified results
        top_results = []
        for doc in docs:
            # Format: "ISBN ReviewText..."
            # Extract ISBN (first token)
            parts = doc.page_content.strip().split(' ', 1)
            isbn = parts[0]
            
            # Lookup Title
            title = isbn_map.get(isbn, f"ISBN:{isbn}")
            
            # Truncate for display
            if len(title) > 40:
                title = title[:37] + "..."
            top_results.append(title)
            
        print(f"  -> Found: {top_results}")
        results.append({
            "query": q,
            "type": case["type"],
            "latency_ms": round(duration, 2),
            "top_results": top_results
        })

    # Save Report
    df = pd.DataFrame(results)
    path = "experiments/01_baseline_results.csv"
    df.to_csv(path, index=False)
    print(f"\n💾 Results saved to {path}")
    
    # Print Summary
    print("\n## Baseline Results Summary")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
