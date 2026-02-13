from typing import List, Any

# Using community version to avoid 'BaseBlobParser' version conflict in langchain-chroma/core
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.infra.config import (
    REVIEW_HIGHLIGHTS_TXT,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES_MAX,
    RRF_K,
    SMALL_TO_BIG_OVER_RETRIEVE_FACTOR,
)
from src.infra.utils import setup_logger
from src.data.stores.metadata_store import metadata_store
from src.data.stores.online_books_store import online_books_store


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
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully")

            if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
                logger.info(f"Loading existing vector database from {CHROMA_DB_DIR}")
                self.db = Chroma(
                    persist_directory=str(CHROMA_DB_DIR),
                    embedding_function=self.embeddings,
                )
                logger.info(
                    "Loaded %s documents from vector database",
                    self.db._collection.count(),
                )
            else:
                error_msg = (
                    f"Vector Database not found at {CHROMA_DB_DIR}.\n"
                    "Please run the initialization script first to build the index:\n"
                    "    python data/scripts/init_db.py"
                )
                logger.warning(error_msg)
                self.db = None

        except Exception as e:
            logger.error("VectorDB init failed [%s]: %s", type(e).__name__, e)
            # Graceful degradation: search() will return []
            self.db = None

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
                logger.warning(
                    "VectorDB: SQLite connection not available. FTS5 disabled."
                )
                self.fts_enabled = False
        except Exception as e:
            logger.error("Failed to initialize FTS5: %s", e)
            self.fts_enabled = False

    def _sparse_fts_search(self, query: str, k: int = 5) -> List[Any]:
        """
        Sparse retrieval: main FTS5 + online staging FTS5. No lock on main DB from writes.
        """
        if not self.fts_enabled:
            logger.warning("FTS5 not enabled, cannot perform sparse search.")
            return []

        class MockDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata

        def mk_doc(row: dict) -> MockDoc:
            title = row.get("title", "") or ""
            desc = row.get("description", "") or ""
            return MockDoc(
                f"{title} {desc}",
                {
                    "isbn": row.get("isbn13", ""),
                    "title": title,
                    "authors": row.get("authors", ""),
                    "categories": row.get("simple_categories", ""),
                },
            )

        results: List[Any] = []
        try:
            # 1. Main store (read-only, no contention)
            conn = metadata_store.connection
            if conn:
                clean_query = query.strip().replace('"', '""')
                if clean_query:
                    fts_query = f'"{clean_query}"'
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT isbn13, title, description, authors, simple_categories
                        FROM books_fts WHERE books_fts MATCH ? ORDER BY rank LIMIT ?
                        """,
                        (fts_query, k),
                    )
                    for row in cursor.fetchall():
                        results.append(mk_doc(dict(row)))

            # 2. Online staging store (separate DB)
            for row in online_books_store.fts_search(query, k=k):
                results.append(mk_doc(row))

            logger.info(
                "VectorDB: FTS5 keyword search found %d results.", len(results)
            )
            return results

        except Exception as e:
            logger.error("VectorDB: FTS5 keyword search failed: %s", e)
            return []

    def search(self, query: str, k: int = 5) -> List[Any]:
        """
        Legacy semantic search.
        """
        if not self.db:
            return []
        return self.db.similarity_search(query, k=k)

    def get_book_details(self, isbn: str):
        """Get book metadata by ISBN from the centralized MetadataStore."""
        return metadata_store.get_book_metadata(str(isbn))

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        rerank: bool = False,
        temporal: bool = False,
    ) -> List[Any]:
        """
        Hybrid Search = Dense (Vector) + Sparse (FTS5) with Reciprocal Rank Fusion (RRF).
        Optional: Cross-Encoder Reranking for high precision.
        """
        if not self.db or not self.fts_enabled:
            logger.warning("FTS5 or DB missing, falling back to simple search.")
            return self.search(query, k)

        # 1. Sparse Retrieval (FTS5)
        # Get top K*2 candidates
        sparse_results = self._sparse_fts_search(query, k=k * 2)

        # Optimization: If alpha=1.0, return Sparse results directly (Skip Dense)
        if alpha == 1.0:
            return sparse_results[:k]

        # 2. Dense Retrieval (Chroma)
        dense_results = self.search(query, k=k * 2)

        # 3. Reciprocal Rank Fusion: score = 1 / (RRF_K + rank)
        fusion_scores = {}

        # Helper to get ID (using ISBN as unique ID)
        def get_id(doc):
            # Try metadata, fallback to content parsing if needed
            if "isbn" in doc.metadata and doc.metadata["isbn"]:
                return str(doc.metadata["isbn"])
            if "isbn13" in doc.metadata and doc.metadata["isbn13"]:
                return str(doc.metadata["isbn13"])

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
            fusion_scores[doc_id]["score"] += 1 / (rank + RRF_K)

        # Fusion: Sparse
        for rank, doc in enumerate(sparse_results):
            doc_id = get_id(doc)
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {"score": 0.0, "doc": doc}
            fusion_scores[doc_id]["score"] += 1 / (rank + RRF_K)

        # Sort by RRF score
        sorted_docs = sorted(
            fusion_scores.values(), key=lambda x: x["score"], reverse=True
        )
        # Keep all unique candidates for reranking
        top_candidates = [item["doc"] for item in sorted_docs]

        # 4. Reranking (Cross-Encoder)
        final_results = top_candidates[:k]
        if rerank:
            from src.rag.reranker import reranker

            rerank_candidates = top_candidates[
                : min(len(top_candidates), RERANK_CANDIDATES_MAX)
            ]
            logger.info("Reranking top %d candidates...", len(rerank_candidates))
            final_results = reranker.rerank(query, rerank_candidates, top_k=k)

        # 5. Temporal Dynamics (Optional)
        # Apply boost to 'final_results' (which now have scores from reranker)
        if temporal:
            from src.rag.temporal import temporal_ranker

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

            final_results = temporal_ranker.apply_decay(final_results, candidate_years)

        return final_results

    def small_to_big_search(self, query: str, k: int = 5) -> List[Any]:
        """
        Small-to-Big Retrieval (Parent-Child / Chunk-to-Parent Pattern).

        ALGORITHM:
          1. Search FINE-GRAINED chunks (e.g. review snippets) for high-precision
             semantic match. User queries like "twist ending" match chunk content.
          2. Map matched chunks back to PARENT books via parent_isbn metadata.
          3. Return full book documents (title, description) for LLM context.

        WHY: Book-level vectors often miss query-specific details (e.g. "unreliable
        narrator"). Chunk-level search finds those, then we lift to book-level.

        Ref: LlamaIndex Recursive Retrieval, RAPTOR (Sarthi et al., 2024).
        """
        from langchain_community.vectorstores import Chroma

        chunk_persist_dir = "data/chroma_chunks"

        # Load chunk index (lazy)
        if not hasattr(self, "chunk_db"):
            try:
                self.chunk_db = Chroma(
                    persist_directory=chunk_persist_dir,
                    embedding_function=self.embeddings,
                    collection_name="review_chunks",
                )
                logger.info("Loaded chunk index from %s", chunk_persist_dir)
            except Exception as e:
                logger.warning(
                    "Chunk index not available: %s. Falling back to hybrid search.", e
                )
                return self.hybrid_search(query, k=k, rerank=True)

        # Step 1: Search chunks (Fine-grained). Over-retrieve to improve recall.
        chunk_results = self.chunk_db.similarity_search(
            query, k=k * SMALL_TO_BIG_OVER_RETRIEVE_FACTOR
        )
        logger.info("Small-to-Big: Found %d chunk matches", len(chunk_results))

        # Step 2: Extract unique parent ISBNs
        parent_isbns = []
        seen = set()
        for chunk in chunk_results:
            isbn = chunk.metadata.get("parent_isbn")
            if isbn and isbn not in seen:
                parent_isbns.append(isbn)
                seen.add(isbn)

        logger.info("Small-to-Big: Mapped to %d unique books", len(parent_isbns))

        # Step 3: Fetch full book context from parent index
        from langchain_core.documents import Document

        parent_docs = []
        for isbn in parent_isbns[:k]:
            rec = metadata_store.get_book_metadata(isbn)
            if rec:
                doc = Document(
                    page_content=(
                        f"Title: {rec.get('title', 'Unknown')}\n"
                        f"ISBN: {rec.get('isbn13', isbn)}\n"
                        f"Description: {rec.get('description', '')}"
                    ),
                    metadata={
                        "isbn": rec.get("isbn13", isbn),
                        "title": rec.get("title"),
                    },
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
            logger.warning(
                "Parent lookup failed, returning chunks with ISBN context"
            )
            # Enrich chunks with their ISBN context
            for chunk in chunk_results[:k]:
                chunk.metadata["note"] = (
                    f"From review of ISBN: {chunk.metadata.get('parent_isbn')}"
                )
            return chunk_results[:k]

        return parent_docs

    def add_book(self, book_data: dict):
        """
        Dynamically add a new book to the ChromaDB vector index.

        Note: FTS5 incremental updates are handled separately via
        metadata_store.insert_book_with_fts() called from RecommendationOrchestrator.add_new_book().
        This method only handles the dense vector index.

        Args:
            book_data: Dict with isbn13, title, authors, description, etc.
        """
        from langchain_core.documents import Document

        isbn = str(book_data.get("isbn13"))
        title = book_data.get("title", "")
        author = book_data.get("authors", "")
        description = book_data.get("description", "")

        # Add to ChromaDB (dense vector index)
        content = (
            f"Title: {title}\nAuthor: {author}\nDescription: {description}\nISBN: {isbn}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "isbn": isbn,
                "isbn13": isbn,
                "title": title,
                "authors": author,
                "description": description,
            },
        )

        if self.db:
            self.db.add_documents([doc])
            logger.info("Added book %s to ChromaDB", isbn)


__all__ = ["VectorDB"]

