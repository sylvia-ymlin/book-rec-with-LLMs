from typing import List, Any
# Using community version to avoid 'BaseBlobParser' version conflict in langchain-chroma/core
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from src.config import REVIEW_HIGHLIGHTS_TXT, CHROMA_DB_DIR, EMBEDDING_MODEL
from src.utils import setup_logger

logger = setup_logger(__name__)


class VectorDB:
    """
    Singleton wrapper for the ChromaDB vector database.
    Uses local sentence-transformers for embedding generation.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDB, cls).__new__(cls)
            cls._instance.db = None
            cls._instance.embeddings = None
        return cls._instance

    def __init__(self):
        if self.db is None:
            self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB with local embeddings."""
        try:
            # Use local sentence-transformers model (no API calls)
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully")

            if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
                logger.info(f"Loading existing vector database from {CHROMA_DB_DIR}")
                self.db = Chroma(
                    persist_directory=str(CHROMA_DB_DIR),
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded {self.db._collection.count()} documents from vector database")
            else:
                error_msg = (
                    f"Vector Database not found at {CHROMA_DB_DIR}.\n"
                    "Please run the initialization script first to build the index:\n"
                    "    python src/init_db.py"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
        except Exception as e:
            logger.error(f"Error initializing Vector DB: {str(e)}")
            self.db = None # Prevent crash if generic error
        
        # 2. Initialize BM25
        self._init_bm25()
        
        # 3. Load Temporal Data (Side-load)
        self.pub_years = {}
        try:
            import pandas as pd
            temporal_df = pd.read_csv("data/books_basic_info.csv", usecols=["isbn13", "publishedDate", "isbn10"])
            # Normalize dates
            from src.core.temporal import temporal_ranker
            
            # Helper to fill dict
            for _, row in temporal_df.iterrows():
                y = temporal_ranker.parse_year(row.get("publishedDate"))
                if y > 0:
                    # Map both IDs if present
                    if pd.notna(row.get("isbn13")):
                        self.pub_years[str(row["isbn13"]).strip().replace(".0", "")] = y
                    if pd.notna(row.get("isbn10")):
                        self.pub_years[str(row["isbn10"]).strip()] = y
                        
            logger.info(f"Loaded {len(self.pub_years)} publication dates for Temporal Scoring.")
        except Exception as e:
            logger.warning(f"Failed to load temporal data: {e}")

    def _init_bm25(self):
        """
        Initialize BM25 index for Sparse (Keyword) Retrieval.
        We load the raw data from CSV because it's faster than iterating Chroma.
        """
        try:
            import pandas as pd
            from rank_bm25 import BM25Okapi
            from src.config import PROCESSED_DATA_DIR
            
            logger.info("Initializing BM25 Index (Sparse Retrieval)...")
            csv_path = PROCESSED_DATA_DIR / "books_processed.csv"
            
            if not csv_path.exists():
                logger.warning("books_processed.csv not found. Hybrid search will be disabled.")
                self.bm25 = None
                return

            df = pd.read_csv(csv_path).fillna("")
            
            # Normalize column names for consistency
            if 'isbn13' in df.columns:
                df['isbn'] = df['isbn13'].astype(str)
            else:
                df['isbn'] = df['isbn'].astype(str)

            # Create corpus: Title + Authors + Description + ISBN
            self.bm25_corpus = df.apply(
                lambda x: f"{x['isbn']} {x['title']} {x['authors']} {x['description']}", axis=1
            ).tolist()
            
            # Basic tokenization
            tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_docs = df.to_dict('records') # Keep raw data for retrieval
            
            # Map for fast lookup
            self.book_map = {str(rec['isbn']): rec for rec in self.bm25_docs}
            
            logger.info(f"BM25 Index built with {len(self.bm25_corpus)} documents.")
            
        except Exception as e:
            logger.error(f"Failed to init BM25: {e}")
            self.bm25 = None

    def search(self, query: str, k: int = 5) -> List[Any]:
        """
        Legacy semantic search.
        """
        if not self.db:
            return []
        return self.db.similarity_search(query, k=k)

    def get_book_details(self, isbn: str):
        """Get book metadata by ISBN"""
        if hasattr(self, 'book_map'):
            return self.book_map.get(str(isbn))
        return None

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5, rerank: bool = False, temporal: bool = False) -> List[Any]:
        """
        Hybrid Search = Dense (Vector) + Sparse (BM25) with Reciprocal Rank Fusion (RRF).
        Optional: Cross-Encoder Reranking for high precision.
        """
        if not self.db or not getattr(self, 'bm25', None):
            logger.warning("BM25 or DB missing, falling back to simple search.")
            return self.search(query, k)

        # 1. Sparse Retrieval (BM25)
        # Get top K*2 candidates
        tokenized_query = query.lower().split()
        # Get scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        # Get top indices
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k*2]
        
        sparse_results = []
        for i in top_n_indices:
            # Structuring like a Document for consistency
            rec = self.bm25_docs[i]
            # Construct page_content similar to what dense loader does
            content = f"Title: {rec.get('title')}\nAuthor: {rec.get('authors')}\nDescription: {rec.get('description')}\nISBN: {rec.get('isbn')}"
            from langchain_core.documents import Document
            doc = Document(page_content=content, metadata=rec)
            doc = Document(page_content=content, metadata=rec)
            sparse_results.append(doc)

        # Optimization: If alpha=1.0, return Sparse results directly (Skip Dense)
        if alpha == 1.0:
            return sparse_results[:k]

        # 2. Dense Retrieval (Chroma)
        dense_results = self.search(query, k=k*2)

        # 3. Reciprocal Rank Fusion
        # Score = 1 / (rank + 60)
        fusion_scores = {}
        
        # Helper to get ID (using ISBN as unique ID)
        def get_id(doc):
            # Try metadata, fallback to content parsing if needed
            if 'isbn' in doc.metadata and doc.metadata['isbn']:
                return str(doc.metadata['isbn'])
            elif 'isbn13' in doc.metadata and doc.metadata['isbn13']:
                 return str(doc.metadata['isbn13'])

            # Fallback parsing
            if "ISBN:" in doc.page_content:
                return doc.page_content.split("ISBN:")[1].strip().split()[0]
            # Fallback 2: Check first word (legacy format)
            return doc.page_content.strip().split()[0]

        # Fusion: Dense
        for rank, doc in enumerate(dense_results):
            doc_id = get_id(doc)
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {"score": 0.0, "doc": doc}
            fusion_scores[doc_id]["score"] += 1 / (rank + 60)

        # Fusion: Sparse
        for rank, doc in enumerate(sparse_results):
            doc_id = get_id(doc)
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {"score": 0.0, "doc": doc}
            fusion_scores[doc_id]["score"] += 1 / (rank + 60)

        # Sort by RRF score
        sorted_docs = sorted(fusion_scores.values(), key=lambda x: x["score"], reverse=True)
        top_candidates = [item["doc"] for item in sorted_docs] # Keep all unique candidates for reranking
        
        # 4. Reranking (Cross-Encoder)
        final_results = top_candidates[:k]
        if rerank:
            from src.core.reranker import reranker
            # Rerank the top 20 (or more) candidates from fusion
            rerank_candidates = top_candidates[:max(k*4, 20)]
            logger.info(f"Reranking top {len(rerank_candidates)} candidates...")
            final_results = reranker.rerank(query, rerank_candidates, top_k=k)

        # 5. Temporal Dynamics (Optional)
        # Apply boost to 'final_results' (which now have scores from reranker)
        if temporal:
            from src.core.temporal import temporal_ranker
            logger.info("Applying Temporal Decay...")
            final_results = temporal_ranker.apply_decay(
                final_results, 
                self.pub_years
            )
            
        return final_results

    def small_to_big_search(self, query: str, k: int = 5) -> List[Any]:
        """
        Small-to-Big Retrieval (Parent-Child Pattern).
        
        SOTA Reference: LlamaIndex Recursive Retrieval, RAPTOR (Sarthi et al., 2024)
        
        1. Search fine-grained review chunks for high precision matching.
        2. Map matched chunks back to their parent books.
        3. Return full book context for LLM.
        """
        from langchain_community.vectorstores import Chroma
        
        CHUNK_PERSIST_DIR = "data/chroma_chunks"
        
        # Load chunk index (lazy)
        if not hasattr(self, 'chunk_db'):
            try:
                self.chunk_db = Chroma(
                    persist_directory=CHUNK_PERSIST_DIR,
                    embedding_function=self.embeddings,
                    collection_name="review_chunks"
                )
                logger.info(f"Loaded chunk index from {CHUNK_PERSIST_DIR}")
            except Exception as e:
                logger.warning(f"Chunk index not available: {e}. Falling back to hybrid search.")
                return self.hybrid_search(query, k=k, rerank=True)
        
        # Step 1: Search chunks (Fine-grained)
        chunk_results = self.chunk_db.similarity_search(query, k=k * 3)  # Over-retrieve
        logger.info(f"Small-to-Big: Found {len(chunk_results)} chunk matches")
        
        # Step 2: Extract unique parent ISBNs
        parent_isbns = []
        seen = set()
        for chunk in chunk_results:
            isbn = chunk.metadata.get("parent_isbn")
            if isbn and isbn not in seen:
                parent_isbns.append(isbn)
                seen.add(isbn)
        
        logger.info(f"Small-to-Big: Mapped to {len(parent_isbns)} unique books")
        
        # Step 3: Fetch full book context from parent index
        # Use BM25 search with ISBN as query (more reliable than metadata filter)
        from langchain_core.documents import Document
        parent_docs = []
        for isbn in parent_isbns[:k]:
            # Search for ISBN in content (books are indexed with ISBN in page_content)
            if getattr(self, 'bm25', None) and getattr(self, 'bm25_docs', None):
                # Use BM25 for exact match
                tokenized_query = isbn.lower().split()
                scores = self.bm25.get_scores(tokenized_query)
                top_idx = scores.argmax()
                if scores[top_idx] > 0:
                    # Convert dict to Document
                    rec = self.bm25_docs[top_idx]
                    doc = Document(
                        page_content=f"Title: {rec.get('title', 'Unknown')}\nISBN: {rec.get('isbn13', isbn)}\nDescription: {rec.get('description', '')}",
                        metadata={"isbn": rec.get('isbn13', isbn), "title": rec.get('title')}
                    )
                    parent_docs.append(doc)
        
        # Fallback: If BM25 didn't work, try similarity search with ISBN
        if not parent_docs and self.db:
            for isbn in parent_isbns[:k]:
                results = self.db.similarity_search(isbn, k=1)
                if results:
                    parent_docs.append(results[0])
        
        # Final fallback: Return chunks with enriched context
        if not parent_docs:
            logger.warning("Parent lookup failed, returning chunks with ISBN context")
            # Enrich chunks with their ISBN context
            for chunk in chunk_results[:k]:
                chunk.metadata["note"] = f"From review of ISBN: {chunk.metadata.get('parent_isbn')}"
            return chunk_results[:k]
        
        return parent_docs

