import os
import shutil
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import REVIEW_HIGHLIGHTS_TXT, CHROMA_DB_DIR, EMBEDDING_MODEL
from tqdm import tqdm

def init_db():
    print("="*50)
    print("📚 Book Recommender: Vector Database Builder")
    print("="*50)
    
    # FIX: Disable Tokenizers Parallelism to prevent deadlocks on macOS
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Force CPU for data ingestion to avoid MPS (Metal) async hangs during long processing.
    # Reliability is key for building the DB; GPU acceleration is only needed for inference.
    device = "cpu"
    print("🐢  Forcing CPU for stable database ingestion (prevents macOS Freezes).")

    # 1. Clear existing DB if any (to avoid duplicates/corruption)
    if CHROMA_DB_DIR.exists():
        print(f"🗑️  Cleaning existing database at {CHROMA_DB_DIR}...")
        shutil.rmtree(CHROMA_DB_DIR)
    
    # 2. Initialize Embeddings
    print(f"🔌 Loading Embedding Model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 512}  # Increase inference batch size for GPU
    )
    
    # 3. Create DB Client
    print(f"💾 Initializing ChromaDB persistence at {CHROMA_DB_DIR}...")
    db = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings
    )
    
    # 4. Stream and Index
    if not REVIEW_HIGHLIGHTS_TXT.exists():
        print(f"❌ Error: Review Highlights file not found at {REVIEW_HIGHLIGHTS_TXT}")
        return

    # Count lines first for progress bar
    print("📊 Counting documents...")
    total_lines = sum(1 for _ in open(REVIEW_HIGHLIGHTS_TXT, 'r', encoding='utf-8'))
    print(f"   Found {total_lines} documents to index.")
    
    batch_size = 2000  # Increased batch size for optimal GPU throughput
    documents = []
    
    # MAX_DOCS=0 for full index; default 20000 for demo
    max_docs = int(os.getenv("MAX_DOCS", "20000")) or None
    print(f"🚀 Starting Ingestion (Source: Review Highlights, Limit: {max_docs or 'all'})...")
    with open(REVIEW_HIGHLIGHTS_TXT, 'r', encoding='utf-8') as f:
        # Use islice for efficient subsetting
        from itertools import islice
        total = min(total_lines, max_docs) if max_docs else total_lines
        for line in tqdm(islice(f, max_docs), total=total, unit="doc", desc="Indexing Reviews"):
            line = line.strip()
            if not line: 
                continue
            
            # Create Document object
            # Note: We assume the line is the ISBN + Description format from previous ETL
            # If strictly just description, simpler. Adapting to generic line-based doc.
            documents.append(Document(page_content=line))
            
            # Batch Insert
            if len(documents) >= batch_size:
                db.add_documents(documents)
                documents = []
                
    # Final Batch
    if documents:
        db.add_documents(documents)
        
    print("\n✅ Verification:")
    print(f"   Total Documents in DB: {db._collection.count()}")
    print("🎉 Vector Database Built Successfully!")

if __name__ == "__main__":
    init_db()
