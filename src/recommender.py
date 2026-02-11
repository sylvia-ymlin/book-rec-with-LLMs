import pandas as pd
from typing import List, Dict, Any
from src.etl import load_books_data
from src.vector_db import VectorDB
from src.config import TOP_K_INITIAL, TOP_K_FINAL
from src.cache import CacheManager

from src.utils import setup_logger, summarize_description
from src.cover_fetcher import fetch_book_cover
from src.marketing.personalized_highlight import get_persona_and_highlights

logger = setup_logger(__name__)

class BookRecommender:
    """
    Core business logic for the Book Recommendation System.
    
    Attributes:
        books (pd.DataFrame): The dataset containing book metadata and emotions.
        vector_db (VectorDB): The vector database instance for semantic search.
        cache (CacheManager): Redis cache manager.
    """
    def __init__(self) -> None:
        """Initialize the recommender by loading data and the vector database."""
        self.books = load_books_data()
        logger.info(f"Loaded books DataFrame with columns: {self.books.columns.tolist()}")
        self.vector_db = VectorDB()
        self.cache = CacheManager()
        # 加载 books_data.csv 的图片链接（只加载 isbn 和 image 字段）
        import pandas as pd
        try:
            # 用新生成的基础表
            book_data_df = pd.read_csv("data/books_basic_info.csv")
            # 统一isbn10为字符串
            book_data_df["isbn10"] = book_data_df["isbn10"].astype(str).str.strip()
            self.book_images = book_data_df.set_index("isbn10")["image"].to_dict()
            self.book_descriptions = book_data_df.set_index("isbn10")["description"].to_dict()
            if "authors" in book_data_df.columns:
                self.book_authors = book_data_df.set_index("isbn10")["authors"].to_dict()
            else:
                self.book_authors = {}
            if "average_rating" in book_data_df.columns:
                self.book_ratings = book_data_df.set_index("isbn10")["average_rating"].to_dict()
            else:
                self.book_ratings = {}
            logger.info(f"Loaded {len(self.book_images)} book images from books_basic_info.csv")
        except Exception as e:
            logger.error(f"Error loading book images from books_data_with_isbn13.csv: {e}")
            self.book_images = {}
            self.book_descriptions = {}
            self.book_authors = {}
        
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
        
        # 1. Semantic Search
        recs = self.vector_db.search(query, k=TOP_K_INITIAL)
        # Handle potential inconsistent ISBN formats (str vs int)
        books_list = []
        for rec in recs:
            isbn_str = rec.page_content.strip('"').split()[0]
            try:
                    # New dataset IDs might be strings (ASIN) or ints
                    books_list.append(isbn_str) 
            except:
                    continue
        
        # 2. Filter by ISBN (Handle both string and int ISBNs from new dataset)
        # Ensure ISBN column type matches
        book_recs = self.books[self.books["isbn13"].astype(str).isin(books_list)].head(TOP_K_INITIAL)
        
        # 3. Filter by Category
        if category and category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category].head(TOP_K_FINAL)
        else:
            book_recs = book_recs.head(TOP_K_FINAL)
            
        results = []
        try:
            for _, row in book_recs.iterrows():
                from html import unescape
                isbn13 = str(row.get("isbn13", "")).strip()
                
                # --- PERFORMANCE FIX: Use static data, NO LLM call during search ---
                # LLM highlights are generated on-demand when user opens book detail
                
                thumbnail = self.book_images.get(row.get("isbn10", "")) or self.book_images.get(isbn13)
                if not thumbnail or pd.isna(thumbnail) or not str(thumbnail).strip():
                    thumbnail = row.get("thumbnail")
                if not thumbnail or pd.isna(thumbnail) or not str(thumbnail).strip():
                    thumbnail = "/assets/cover-not-found.jpg"
                if isinstance(thumbnail, str) and thumbnail.startswith("/Users/"):
                    thumbnail = "/assets/cover-not-found.jpg"

                average_rating = self.book_ratings.get(row.get("isbn10", "")) or self.book_ratings.get(isbn13) or 0
                tags_raw = str(row.get("tags", "")).strip()
                tags = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []
                emotions = {
                    "joy": float(row.get("joy", 0.0)),
                    "sadness": float(row.get("sadness", 0.0)),
                    "fear": float(row.get("fear", 0.0)),
                    "anger": float(row.get("anger", 0.0)),
                    "surprise": float(row.get("surprise", 0.0)),
                }
                
                # Use static review_highlights from CSV (no LLM)
                review_highlights_raw = str(row.get("review_highlights", ""))
                review_highlights = [h.strip() for h in review_highlights_raw.split(";") if h.strip()][:3]
                
                results.append({
                    "isbn": isbn13,
                    "title": row.get("title", ""),
                    "authors": row.get("authors", "Unknown"),
                    "description": row.get("description", ""),
                    "thumbnail": thumbnail,
                    "caption": f"{row.get('title', '')} by {row.get('authors', 'Unknown')}: {row.get('description', '')}",
                    "tags": tags,
                    "emotions": emotions,
                    "review_highlights": review_highlights,
                    "persona_summary": "",  # Filled on-demand in /marketing/highlights
                    "average_rating": average_rating
                })
            logger.info(f"Sample result: {results[0] if results else 'EMPTY'}")
            return results
        except Exception as e:
            import traceback
            logger.error(f"Error in _format_results: {e}\n{traceback.format_exc()}")
            return []

