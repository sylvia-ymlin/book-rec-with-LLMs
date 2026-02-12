import time
import pandas as pd
from src.core.rag.vector_db import VectorDB
from src.core.rag.router import QueryRouter

def run_benchmark():
    print("🚀 Starting Agentic Router Benchmark...")
    
    # Init Components
    db = VectorDB()
    router = QueryRouter()
    
    # Load Title Mapping (for display)
    try:
        books_df = pd.read_csv("data/books_processed.csv")
        if 'isbn13' in books_df.columns:
            books_df['isbn'] = books_df['isbn13'].astype(str)
        else:
             books_df['isbn'] = books_df['isbn'].astype(str)
        isbn_map = books_df.set_index('isbn')['title'].to_dict()
    except:
        isbn_map = {}

    test_queries = [
        # 1. ISBN -> Should be EXACT (No Rerank) to avoid regression
        {"query": "0060959479", "expected_strat": "exact"},
        
        # 2. Keyword -> Should be FAST (No Rerank)
        {"query": "python programming", "expected_strat": "fast"},
        
        # 3. Complex -> Should be DEEP (With Rerank)
        {"query": "books about finding love in unexpected places", "expected_strat": "deep"},
    ]
    
    results = []
    
    for case in test_queries:
        q = case["query"]
        print(f"\nUser Query: '{q}'")
        
        # 1. ROUTING STEP
        route_decision = router.route(q)
        strat = route_decision["strategy"]
        use_rerank = route_decision["rerank"]
        alpha_val = route_decision.get("alpha", 0.5)
        
        print(f"  🤖 Router Decision: {strat.upper()} (Rerank={use_rerank}, Alpha={alpha_val})")
        
        # Check expectation
        if strat != case["expected_strat"]:
             print(f"  ⚠️ WARNING: Expected {case['expected_strat']}, got {strat}")
        
        # 2. RETRIEVAL STEP
        start_time = time.time()
        docs = db.hybrid_search(
            q, 
            k=5, 
            rerank=use_rerank,
            alpha=alpha_val
        )
        duration = (time.time() - start_time) * 1000
        
        # Capture results
        top_results = []
        for doc in docs:
            # Extract ISBN/Title
            if "ISBN:" in doc.page_content:
                 isbn = doc.page_content.split("ISBN:")[1].strip().split()[0]
            else:
                 parts = doc.page_content.strip().split(' ', 1)
                 isbn = parts[0]

            title = isbn_map.get(isbn, f"ISBN:{isbn}")
            if len(title) > 30:
                title = title[:27] + "..."
            
            score = doc.metadata.get("relevance_score", "N/A")
            if score != "N/A":
                top_results.append(f"{title} ({score:.4f})")
            else:
                top_results.append(f"{title}")
            
        print(f"  -> Found: {top_results[:3]}")
        results.append({
            "query": q,
            "strategy": strat,
            "latency_ms": round(duration, 2),
            "top_1": top_results[0] if top_results else "None"
        })

    # Save
    df = pd.DataFrame(results)
    path = "experiments/04_router_results.csv"
    df.to_csv(path, index=False)
    print(f"\n💾 Results saved to {path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
