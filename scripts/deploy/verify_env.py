
import sys
import os
import platform
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def print_status(component, status, message=""):
    color = "\033[92m" if status == "OK" else "\033[91m"
    reset = "\033[0m"
    print(f"[{component.ljust(15)}] {color}{status}{reset} {message}")

def check_system():
    print("\n=== System Info ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.system()} {platform.release()}")
    print_status("System", "OK")

def check_torch():
    print("\n=== PyTorch Check ===")
    try:
        import torch
        print(f"Torch Version: {torch.__version__}")
        if torch.backends.mps.is_available():
            print_status("Accelerator", "OK", "MPS (Metal Performance Shaders) is available! 🚀")
        elif torch.cuda.is_available():
            print_status("Accelerator", "OK", f"CUDA is available! ({torch.cuda.get_device_name(0)})")
        else:
            print_status("Accelerator", "WARN", "Running on CPU (Slow but safe)")
    except ImportError:
        print_status("PyTorch", "FAIL", "Not installed")

def check_vector_db():
    print("\n=== Vector Database Check ===")
    try:
        from src.rag.vector_db import VectorDB
        
        start = time.perf_counter()
        print("Loading VectorDB (this loads embeddings)...")
        vdb = VectorDB()
        print(f"Load Time: {time.perf_counter() - start:.2f}s")
        
        # Test Query
        test_q = "fantasy"
        results = vdb.search(test_q, k=1)
        if results:
            print_status("ChromaDB", "OK", f"Query '{test_q}' returned: {results[0].page_content[:50]}...")
        else:
            print_status("ChromaDB", "WARN", "Database loaded but returned empty results")
            
    except Exception as e:
        print_status("ChromaDB", "FAIL", str(e))
        print("Tip: Ensure 'data/chroma_db' exists.")

if __name__ == "__main__":
    check_system()
    check_torch()
    check_vector_db()
