import time
import pandas as pd
from pathlib import Path
from src.rag.vector_db import VectorDB


def run_benchmark():
    print("🚀 Starting Hybrid Retrieval Benchmark...")

    # Load Title Mapping
    try:
        books_df = pd.read_csv("data/books_processed.csv")
        # Ensure string ISBN for matching
        if "isbn13" in books_df.columns:
            books_df["isbn"] = books_df["isbn13"].astype(str)
        else:
            books_df["isbn"] = books_df["isbn"].astype(str)

        isbn_map = books_df.set_index("isbn")["title"].to_dict()
    except Exception as e:
        print(f"⚠️ Failed to load books_processed.csv: {e}")
        isbn_map = {}

    db = VectorDB()

    # Same Test Cases
    test_queries = [
        # 1. Semantic (Hybrid should match Dense)
        {"type": "Semantic", "query": "books about finding love in unexpected places"},
        {"type": "Semantic", "query": "scary stories that keep you up at night"},
        # 2. Keyword/Proper Noun (Hybrid should improve)
        {"type": "Keyword", "query": "Harry Potter"},
        {"type": "Keyword", "query": "Python Programming"},
        {"type": "Keyword", "query": "Jane Austen"},
        # 3. Exact Match / ISBN (Hybrid should fix this)
        {"type": "Exact", "query": "0060959479"},
    ]

    results = []

    for case in test_queries:
        q = case["query"]
        print(f"\nScanning: '{q}' ({case['type']})...")

        start_time = time.time()
        # USE HYBRID SEARCH
        docs = db.hybrid_search(q, k=5)
        duration = (time.time() - start_time) * 1000

        # Capture simplified results
        top_results = []
        for doc in docs:
            # Extract ISBN
            parts = doc.page_content.strip().split(" ", 1)
            isbn = parts[0]
            # Fallback parsing for legacy docs
            if "ISBN:" in doc.page_content:
                isbn = doc.page_content.split("ISBN:")[1].strip().split()[0]

            title = isbn_map.get(isbn, f"ISBN:{isbn}")
            if len(title) > 40:
                title = title[:37] + "..."
            top_results.append(title)

        print(f"  -> Found: {top_results}")
        results.append(
            {
                "query": q,
                "type": case["type"],
                "latency_ms": round(duration, 2),
                "top_results": top_results,
            }
        )

    # Save
    df = pd.DataFrame(results)
    path = "benchmarks/results/02_hybrid_results.csv"
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n💾 Results saved to {path}")

    print("\n## Hybrid Search Results")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_benchmark()

