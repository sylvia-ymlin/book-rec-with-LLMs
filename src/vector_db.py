import gc
from typing import List, Any
# Using community version to avoid 'BaseBlobParser' version conflict in langchain-chroma/core
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from src.config import REVIEW_HIGHLIGHTS_TXT, CHROMA_DB_DIR, EMBEDDING_MODEL
from src.utils import setup_logger
from src.core.metadata_store import metadata_store
import sqlite3

logger = setup_logger(__name__)


class VectorDB:
    """
    Hybrid Vector Database combining ChromaDB (Dense) and SQLite FTS5 (Sparse).
    
    ENGINEERING IMPROVEMENT:
    Transitioned from in-memory `rank_bm25` logic to a disk-based SQLite FTS5 
    architecture for keyword search. This allows for zero-RAM search indexing and 
    eliminates the need for dataset pruning.
    
    Features:
    - Zero-RAM Keyword Indexing (via FTS5).
    - Hybrid RRF scoring (ChromaDB + FTS5).
    - Persistence on disk for 221k+ items.
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
                logger.warning(error_msg)
                self.db = None
                
        except Exception as e:
            logger.error(f"Error initializing Vector DB: {str(e)}")
            self.db = None # Prevent crash if generic error
        
        # 2. Initialize FTS5 (Sparse Retrieval)
        self._init_fts5()
        
        # 3. Initialize Temporal Data logic (Zero-RAM mode)
        logger.info("VectorDB: Temporal Scoring will use SQLite metadata.")

    def _init_fts5(self):
        """
        Initialize FTS5 for Sparse (Keyword) Retrieval via SQLite.
        """
        try:
            conn = metadata_store.connection
            if conn:
                logger.info("VectorDB: FTS5 keyword search enabled via SQLite.")
                self.fts_enabled = True
            else:
                logger.warning("VectorDB: SQLite connection not available. FTS5 disabled.")
                self.fts_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize FTS5: {e}")
            self.fts_enabled = False

    def _sparse_fts_search(self, query: str, k: int = 5) -> List[Any]:
        """
        Performs sparse retrieval using SQLite FTS5.
        """
        if not self.fts_enabled:
            logger.warning("FTS5 not enabled, cannot perform sparse search.")
            return []

        try:
            conn = metadata_store.connection
            if not conn:
                logger.warning("VectorDB: SQLite connection not available. Keyword search disabled.")
                return []

            # FTS5 Full Text Search
            query_sql = """
                SELECT isbn13, title, description, authors, simple_categories, rank
                FROM books_fts
                WHERE books_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            
            # Clean query for FTS5 (escape special chars)
            clean_query = query.strip().replace('"', '""')
            if not clean_query: return []
            
            # Prepare query for prefix search if needed
            fts_query = f'"{clean_query}"'
            
            cursor = conn.cursor()
            cursor.execute(query_sql, (fts_query, k))
            rows = cursor.fetchall()

            class MockDoc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata

            results = []
            for row in rows:
                content = f"{row['title']} {row['description']}"
                metadata = {
                    "isbn": row["isbn13"],
                    "title": row["title"],
                    "authors": row["authors"],
                    "categories": row["simple_categories"]
                }
                results.append(MockDoc(content, metadata))

            logger.info(f"VectorDB: FTS5 keyword search found {len(results)} results.")
            return results

        except Exception as e:
            logger.error(f"VectorDB: FTS5 keyword search failed: {e}")
            return []

    def search(self, query: str, k: int = 5) -> List[Any]:
        """
        Legacy semantic search.
        """
        if not self.db:
            return []
        return self.db.similarity_search(query, k=k)

    def get_book_details(self, isbn: str):
        """Get book metadata by ISBN"""
        # This method might need to be updated to query metadata_store directly
        # For now, it's left as is, assuming book_map might be re-introduced or replaced.
        # If book_map is removed, this will always return None.
        return None

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5, rerank: bool = False, temporal: bool = False) -> List[Any]:
        """
        Hybrid Search = Dense (Vector) + Sparse (FTS5) with Reciprocal Rank Fusion (RRF).
        Optional: Cross-Encoder Reranking for high precision.
        """
        if not self.db or not self.fts_enabled:
            logger.warning("FTS5 or DB missing, falling back to simple search.")
            return self.search(query, k)

        # 1. Sparse Retrieval (FTS5)
        # Get top K*2 candidates
        sparse_results = self._sparse_fts_search(query, k=k*2)

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
            
            # Populate local year map for candidates to avoid repeated queries
            candidate_years = {}
            for doc in final_results:
                isbn = get_id(doc)
                if isbn:
                    rec = metadata_store.get_book_metadata(isbn)
                    year = temporal_ranker.parse_year(rec.get("publishedDate"))
                    if year > 0:
                        candidate_years[isbn] = year
            
            final_results = temporal_ranker.apply_decay(
                final_results, 
                candidate_years
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
        from langchain_core.documents import Document
        parent_docs = []
        for isbn in parent_isbns[:k]:
            rec = metadata_store.get_book_metadata(isbn)
            if rec:
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

    def add_book(self, book_data: dict):
        """
        Dynamically add a new book to the vector database and update indices.
        """
        from langchain_core.documents import Document
        
        isbn = str(book_data.get("isbn13"))
        title = book_data.get("title", "")
        author = book_data.get("authors", "")
        description = book_data.get("description", "")
        
        # 1. Add to Chroma
        content = f"Title: {title}\nAuthor: {author}\nDescription: {description}\nISBN: {isbn}"
        doc = Document(
            page_content=content, 
            metadata={
                "isbn": isbn, 
                "isbn13": isbn, 
                "title": title, 
                "authors": author, 
                "description": description
            }
        )
        
        if self.db:
            self.db.add_documents([doc])
            logger.info(f"Added book {isbn} to ChromaDB")
            
        if hasattr(self, 'fts_enabled') and self.fts_enabled:
            logger.info("Note: FTS5 database updates are not implemented in add_book yet.")

