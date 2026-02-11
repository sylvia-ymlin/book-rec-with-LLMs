import pandas as pd
from typing import List, Dict, Any
from src.etl import load_books_data
from src.vector_db import VectorDB
from src.config import TOP_K_INITIAL, TOP_K_FINAL, DATA_DIR
from src.cache import CacheManager

from src.utils import setup_logger, summarize_description
from src.cover_fetcher import fetch_book_cover
from src.marketing.personalized_highlight import get_persona_and_highlights
from src.core.metadata_store import metadata_store

logger = setup_logger(__name__)

class BookRecommender:
    """
    Core Recommendation Engine orchestrating search and metadata enrichment.
    
    ENGINEERING IMPROVEMENT:
    Refactored to be entirely DataFrame-free. All metadata is now fetched 
    on-demand via `MetadataStore` (SQLite), ensuring zero-RAM overhead for 
    candidate enrichment and category filtering.
    
    Attributes:
        books (pd.DataFrame): The dataset containing book metadata and emotions.
        vector_db (VectorDB): The vector database instance for semantic search.
        cache (CacheManager): Redis cache manager.
    """
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
        results = []
        for isbn in books_list:
            meta = metadata_store.get_book_metadata(isbn)
            if not meta:
                continue
            
            # Category filter
            if category and category != "All":
                if meta.get("simple_categories") != category:
                    continue
            
            # Tone enrichment and basic formatting
            from html import unescape
            
            thumbnail = meta.get("thumbnail")
            if not thumbnail or pd.isna(thumbnail) or not str(thumbnail).strip():
                thumbnail = "/assets/cover-not-found.jpg"
            
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
                "isbn": isbn,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", "Unknown"),
                "description": meta.get("description", ""),
                "thumbnail": thumbnail,
                "caption": f"{meta.get('title', '')} by {meta.get('authors', 'Unknown')}",
                "tags": tags,
                "emotions": emotions,
                "review_highlights": highlights,
                "persona_summary": "",
                "average_rating": float(meta.get("average_rating", 0.0))
            })
            
            if len(results) >= TOP_K_FINAL:
                break
                
        return results

    def get_categories(self) -> List[str]:
        """Get unique book categories from SQLite."""
        return ["All"] + metadata_store.get_all_categories()

    def get_tones(self) -> List[str]:
        """Get available emotional tones."""
        return ["All", "Happy", "Sad", "Fear", "Anger", "Surprise"]

    def add_new_book(self, isbn: str, title: str, author: str, description: str, category: str = "General", thumbnail: str = None) -> Any:
        """
        Add a new book to the system: CSV, Memory, and Vector DB.
        Returns the new book dictionary if successful, None otherwise.
        """
        try:
            import pandas as pd
            
            # 1. Update Persistent Storage (CSV)
            csv_path = DATA_DIR / "books_processed.csv"
            
            # Define new row with all expected columns
            new_row = {
                "isbn13": isbn,
                "title": title,
                "authors": author,
                "description": description,
                "simple_categories": category,
                "thumbnail": thumbnail if thumbnail else "/assets/cover-not-found.jpg",
                "average_rating": 0.0,
                "joy": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, "surprise": 0.0,
                "tags": "", "review_highlights": "",
                "isbn10": str(isbn)[:10] # Approximation
            }
            
            # Check for duplicates in memory first
            isbn_s = str(isbn)
            if isbn_s in self.book_images or (hasattr(self, 'books') and str(isbn) in self.books['isbn13'].astype(str).values):
                logger.warning(f"Book {isbn} already exists. Skipping add.")
                return None
                
            # Append to CSV
            if csv_path.exists():
                # Read just the header to align columns
                header_df = pd.read_csv(csv_path, nrows=0)
                csv_columns = header_df.columns.tolist()
                
                # Filter/Order new_row to match CSV structure
                ordered_row = {}
                for col in csv_columns:
                    ordered_row[col] = new_row.get(col, "") # Default to empty string if missing
                
                # Append to CSV
                pd.DataFrame([ordered_row]).to_csv(csv_path, mode='a', header=False, index=False)
            else:
                 pd.DataFrame([new_row]).to_csv(csv_path, index=False)
                 
            # 2. Update In-Memory DataFrame (self.books)
            # Add 'large_thumbnail' which load_books_data adds
            new_row["large_thumbnail"] = new_row["thumbnail"]
            
            # Append to self.books
            if self.books is not None:
                self.books = pd.concat([self.books, pd.DataFrame([new_row])], ignore_index=True)
            
            # 3. Update In-Memory Lookups (for get_recommendations speed)
            self.book_images[isbn_s] = new_row["thumbnail"]
            self.book_descriptions[isbn_s] = description
            self.book_authors[isbn_s] = author
            self.book_ratings[isbn_s] = 0.0
            
            # 3. Update In-Memory Lookups (for get_recommendations speed)
            self.book_descriptions[str(isbn)] = description
            
            # 4. Update Vector DB (Chroma + BM25)
            self.vector_db.add_book(new_row)
            
            logger.info(f"Successfully added book {isbn}: {title}")
            return new_row
            
        except Exception as e:
            logger.error(f"Error adding new book: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
