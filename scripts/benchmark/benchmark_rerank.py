import time
import pandas as pd
from src.core.rag.vector_db import VectorDB

def run_benchmark():
    print("🚀 Starting Reranked Retrieval Benchmark...")
    
    # Load Title Mapping
    try:
        books_df = pd.read_csv("data/books_processed.csv")
        if 'isbn13' in books_df.columns:
            books_df['isbn'] = books_df['isbn13'].astype(str)
        else:
             books_df['isbn'] = books_df['isbn'].astype(str)
        isbn_map = books_df.set_index('isbn')['title'].to_dict()
    except Exception as e:
        print(f"⚠️ Failed to load books_processed.csv: {e}")
        isbn_map = {}

    db = VectorDB()
    
    # Same Test Cases
    test_queries = [
        # 1. Semantic (Reranker should bubble up best Semantic matches)
        {"type": "Semantic", "query": "books about finding love in unexpected places"},
        # Complex mood query
        {"type": "Complex", "query": "a dark sci-fi thriller with a female protagonist"},
        
        # 2. Keyword/Proper Noun (Reranker should confirm these are relevant)
        {"type": "Keyword", "query": "Harry Potter"}, 
        {"type": "Keyword", "query": "Jane Austen"}, 
        
        # 3. Exact Match (Should still work)
        {"type": "Exact", "query": "0060959479"}, 
    ]
    
    results = []
    
    for case in test_queries:
        q = case["query"]
        print(f"\nScanning: '{q}' ({case['type']})...")
        
        start_time = time.time()
        # USE HYBRID WITH RERANK
        docs = db.hybrid_search(q, k=5, rerank=True)
        duration = (time.time() - start_time) * 1000
        
        # Capture results with scores
        top_results = []
        for doc in docs:
            # Extract ISBN
            parts = doc.page_content.strip().split(' ', 1)
            isbn = parts[0]
            if "ISBN:" in doc.page_content:
                 isbn = doc.page_content.split("ISBN:")[1].strip().split()[0]

            title = isbn_map.get(isbn, f"ISBN:{isbn}")
            if len(title) > 30:
                title = title[:27] + "..."
            
            score = doc.metadata.get("relevance_score", 0.0)
            top_results.append(f"{title} ({score:.4f})")
            
        print(f"  -> Found: {top_results}")
        results.append({
            "query": q,
            "type": case["type"],
            "latency_ms": round(duration, 2),
            "top_results": top_results
        })

    # Save
    df = pd.DataFrame(results)
    path = "experiments/03_rerank_results.csv"
    df.to_csv(path, index=False)
    print(f"\n💾 Results saved to {path}")
    
    print("\n## Reranked Search Results")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
