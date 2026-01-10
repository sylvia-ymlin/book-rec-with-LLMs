import requests
import time
import numpy as np
import sys

BASE_URL = "http://localhost:6006"

def wait_for_service(timeout=60):
    start = time.time()
    print("Waiting for service to be ready...")
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=1)
            if resp.status_code == 200:
                print(f"Service ready in {time.time() - start:.2f}s")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print("Service timed out.")
    return False

def benchmark_endpoint(name, url, method="GET", json_body=None, n=10, warmup=2):
    print(f"\nBenchmarking {name} ({method} {url})...")
    latencies = []
    
    # Warmup
    for _ in range(warmup):
        try:
            if method == "GET":
                requests.get(url)
            else:
                requests.post(url, json=json_body)
        except:
            pass
            
    # Test
    for i in range(n):
        start = time.perf_counter()
        try:
            if method == "GET":
                resp = requests.get(url)
            else:
                resp = requests.post(url, json=json_body)
            resp.raise_for_status()
            duration = (time.perf_counter() - start) * 1000 # ms
            latencies.append(duration)
            sys.stdout.write(".")
            sys.stdout.flush()
        except Exception as e:
            print(f"E({e})")
            
    print("\n")
    if not latencies:
        print("All requests failed.")
        return

    print(f"Results for {name}:")
    print(f"  Count:  {len(latencies)}")
    print(f"  Mean:   {np.mean(latencies):.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  Min:    {np.min(latencies):.2f} ms")
    print(f"  Max:    {np.max(latencies):.2f} ms")

def main():
    if not wait_for_service(timeout=120): # Longer timeout for model loading
        sys.exit(1)
        
    # 1. Personalized Recommendations (Cold -> Warm)
    benchmark_endpoint(
        "Personalized Recs (Cached/Computed)", 
        f"{BASE_URL}/api/recommend/personal?user_id=local&top_k=20",
        n=20
    )
    
    # 2. Search (Vector DB)
    benchmark_endpoint(
        "Semantic Search 'machine learning'", 
        f"{BASE_URL}/recommend", 
        method="POST",
        json_body={"query": "machine learning", "category": "All", "tone": "All"},
        n=20
    )
    
    # 3. Book Details (Metadata lookup via Favorites list)
    # We simulate this by calling favorites list which does metadata lookups
    benchmark_endpoint(
        "Favorites List (Metadata Lookup)", 
        f"{BASE_URL}/favorites/list/local", 
        n=20
    )

if __name__ == "__main__":
    main()
