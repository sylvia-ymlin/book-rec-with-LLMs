from typing import List, Dict, Any, Optional
from src.vector_db import VectorDB
from src.config import TOP_K_INITIAL, TOP_K_FINAL, DATA_DIR
from src.cache import CacheManager

from src.utils import setup_logger
from src.core.metadata_store import metadata_store

logger = setup_logger(__name__)

class BookRecommender:
    """Orchestrates RAG search and metadata enrichment. Zero-RAM: metadata from SQLite on demand."""
    def __init__(self) -> None:
        """Initialize the recommender by loading data and the vector database."""
        # We no longer load self.books or in-memory maps.
        # Everything is fetched on-demand from MetadataStore (SQLite).
        
        self.vector_db = VectorDB()
        self.cache = CacheManager()
        
        logger.info("BookRecommender: Zero-RAM mode enabled. Using SQLite for on-demand lookups.")
        
    def get_recommendations(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        user_id: str = "local"
    ) -> List[Dict[str, Any]]:
        """
        Generate book recommendations based on query, category, and tone.
        """
        if not query or not query.strip():
            return []

        # Check Cache
        cache_key = self.cache.generate_key("rec", q=query, c=category, t=tone)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached results for key: {cache_key}")
            return cached_result

        logger.info(f"Processing request: query='{query}', category='{category}', tone='{tone}'")
        
        # 1. Agentic Retrieval (Router -> Hybrid/Rerank/Small-to-Big)
        from src.core.router import QueryRouter
        router = QueryRouter()
        decision = router.route(query)
        logger.info(f"Retrieval Strategy: {decision}")
        
        # Route to appropriate search method
        if decision["strategy"] == "small_to_big":
            recs = self.vector_db.small_to_big_search(query, k=TOP_K_INITIAL)
        else:
            recs = self.vector_db.hybrid_search(
                query, 
                k=TOP_K_INITIAL, 
                alpha=decision.get("alpha", 0.5), 
                rerank=decision["rerank"],
                temporal=decision.get("temporal", False)
            )

        books_list = []
        for rec in recs:
            # Robust ISBN Extraction
            isbn_str = None
            
            # 1. Try Metadata (Hybrid/BM25)
            if rec.metadata and 'isbn' in rec.metadata:
                isbn_str = str(rec.metadata['isbn'])
            elif rec.metadata and 'isbn13' in rec.metadata:
                isbn_str = str(rec.metadata['isbn13'])
            
            # 2. Try New Content Format (Title... ISBN: X)
            elif "ISBN:" in rec.page_content:
                try:
                    # Find 'ISBN:' and take next token
                    parts = rec.page_content.split("ISBN:")
                    if len(parts) > 1:
                        isbn_str = parts[1].strip().split()[0]
                except:
                    pass

            # 3. Try Legacy Content Format (Start of string)
            if not isbn_str:
                isbn_str = rec.page_content.strip('"').split()[0]
            
            if isbn_str:
                books_list.append(isbn_str)
        
        # 2. Enrich and Format results (Zero-RAM mode)
        from src.utils import enrich_book_metadata  # Use centralized logic
        
        results = []
        for isbn in books_list:
            meta = metadata_store.get_book_metadata(str(isbn))
            
            # Enrich with dynamic cover fetching if needed
            meta = enrich_book_metadata(meta, str(isbn))
            
            if not meta:
                continue
            
            # Category filter
            if category and category != "All":
                if meta.get("simple_categories") != category:
                    continue
            
            # Tone enrichment and basic formatting
            from html import unescape
            
            thumbnail = meta.get("thumbnail")
            
            tags_raw = str(meta.get("tags", "")).strip()
            tags = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []
            
            emotions = {
                "joy": float(meta.get("joy", 0.0)),
                "sadness": float(meta.get("sadness", 0.0)),
                "fear": float(meta.get("fear", 0.0)),
                "anger": float(meta.get("anger", 0.0)),
                "surprise": float(meta.get("surprise", 0.0)),
            }
            
            highlights_raw = str(meta.get("review_highlights", ""))
            highlights = [h.strip() for h in highlights_raw.split(";") if h.strip()][:3]
            
            results.append({
                "isbn": str(isbn),
                "title": meta.get("title", ""),
                "authors": meta.get("authors", "Unknown"),
                "description": meta.get("description", ""),
                "thumbnail": thumbnail,
                "caption": f"{meta.get('title', '')} by {meta.get('authors', 'Unknown')}",
                "tags": tags,
                "emotions": emotions,
                "review_highlights": highlights,
                "persona_summary": "",
                "average_rating": float(meta.get("average_rating", 0.0)),
                "source": "local",  # Track data source
            })
            
            if len(results) >= TOP_K_FINAL:
                break
        
        # 3. Web Search Fallback (Freshness-Aware)
        # Triggered when: freshness_fallback=True AND local results < threshold
        if decision.get("freshness_fallback", False):
            threshold = decision.get("freshness_threshold", 3)
            if len(results) < threshold:
                web_results = self._fetch_from_web(query, TOP_K_FINAL - len(results), category)
                results.extend(web_results)
                logger.info(f"Web fallback added {len(web_results)} books")
        
        # Cache the results
        if results:
            self.cache.set(cache_key, results)
                
        return results
    
    def _fetch_from_web(
        self, 
        query: str, 
        max_results: int,
        category: str = "All"
    ) -> List[Dict[str, Any]]:
        """
        Fetch books from Google Books API when local results are insufficient.
        Auto-persists discovered books to local database for future queries.
        
        Args:
            query: User's search query
            max_results: Maximum number of results to fetch
            category: Category filter (not applied to web search, used for filtering results)
        
        Returns:
            List of formatted book dicts ready for response
        """
        try:
            from src.core.web_search import search_google_books
        except ImportError:
            logger.warning("Web search module not available")
            return []
        
        results = []
        
        try:
            web_books = search_google_books(query, max_results=max_results * 2)
            
            for book in web_books:
                isbn = book.get("isbn13", "")
                if not isbn:
                    continue
                
                # Skip if already in local database
                if metadata_store.book_exists(isbn):
                    continue
                
                # Category filter (if specified)
                if category and category != "All":
                    book_cat = book.get("simple_categories", "")
                    if category.lower() not in book_cat.lower():
                        continue
                
                # Auto-persist to local database
                added = self.add_new_book(
                    isbn=isbn,
                    title=book.get("title", ""),
                    author=book.get("authors", "Unknown"),
                    description=book.get("description", ""),
                    category=book.get("simple_categories", "General"),
                    thumbnail=book.get("thumbnail"),
                    published_date=book.get("publishedDate", ""),
                )
                
                if added:
                    results.append({
                        "isbn": isbn,
                        "title": book.get("title", ""),
                        "authors": book.get("authors", "Unknown"),
                        "description": book.get("description", ""),
                        "thumbnail": book.get("thumbnail", ""),
                        "caption": f"{book.get('title', '')} by {book.get('authors', 'Unknown')}",
                        "tags": [],
                        "emotions": {"joy": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, "surprise": 0.0},
                        "review_highlights": [],
                        "persona_summary": "",
                        "average_rating": float(book.get("average_rating", 0.0)),
                        "source": "google_books",  # Track data source
                    })
                
                if len(results) >= max_results:
                    break
            
            logger.info(f"Web fallback: Found and persisted {len(results)} new books")
            return results
            
        except Exception as e:
            logger.error(f"Web fallback failed: {e}")
            return []

    def get_categories(self) -> List[str]:
        """Get unique book categories from SQLite."""
        return ["All"] + metadata_store.get_all_categories()

    def get_tones(self) -> List[str]:
        """Get available emotional tones."""
        return ["All", "Happy", "Sad", "Fear", "Anger", "Surprise"]

    def add_new_book(
        self, 
        isbn: str, 
        title: str, 
        author: str, 
        description: str, 
        category: str = "General", 
        thumbnail: Optional[str] = None,
        published_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add a new book to the system: CSV, SQLite (with FTS5), and ChromaDB.
        
        Args:
            isbn: ISBN-13 or ISBN-10
            title: Book title
            author: Author name(s)
            description: Book description
            category: Book category
            thumbnail: Cover image URL
            published_date: Publication date (YYYY, YYYY-MM, or YYYY-MM-DD)
        
        Returns:
            New book dictionary if successful, None otherwise
        """
        try:
            import pandas as pd
            
            isbn_s = str(isbn).strip()
            
            # Check if already exists
            if metadata_store.book_exists(isbn_s):
                logger.debug(f"Book {isbn} already exists. Skipping add.")
                return None
            
            # 1. Update Persistent Storage (CSV)
            csv_path = DATA_DIR / "books_processed.csv"
            
            # Define new row with all expected columns
            new_row = {
                "isbn13": isbn_s,
                "title": title,
                "authors": author,
                "description": description,
                "simple_categories": category,
                "thumbnail": thumbnail if thumbnail else "/assets/cover-not-found.jpg",
                "average_rating": 0.0,
                "joy": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, "surprise": 0.0,
                "tags": "", "review_highlights": "",
                "isbn10": isbn_s[:10] if len(isbn_s) >= 10 else isbn_s,
                "publishedDate": published_date or "",
                "source": "google_books",  # Track data source
            }
                
            # Append to CSV
            if csv_path.exists():
                # Read just the header to align columns
                header_df = pd.read_csv(csv_path, nrows=0)
                csv_columns = header_df.columns.tolist()
                
                # Filter/Order new_row to match CSV structure
                ordered_row = {}
                for col in csv_columns:
                    ordered_row[col] = new_row.get(col, "")
                
                # Append to CSV
                pd.DataFrame([ordered_row]).to_csv(csv_path, mode='a', header=False, index=False)
            else:
                pd.DataFrame([new_row]).to_csv(csv_path, index=False)
                 
            new_row["large_thumbnail"] = new_row["thumbnail"]
            new_row["image"] = new_row["thumbnail"]

            # 2. Insert into SQLite with FTS5 (incremental indexing)
            metadata_store.insert_book_with_fts(new_row)

            # 3. Update Vector DB (ChromaDB)
            self.vector_db.add_book(new_row)
            
            logger.info(f"Successfully added book {isbn}: {title}")
            return new_row
            
        except Exception as e:
            logger.error(f"Error adding new book: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
