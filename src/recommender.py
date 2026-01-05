import pandas as pd
from typing import List, Dict, Any
from src.etl import load_books_data
from src.vector_db import VectorDB
from src.config import TOP_K_INITIAL, TOP_K_FINAL
from src.utils import setup_logger

logger = setup_logger(__name__)

class BookRecommender:
    """
    Core business logic for the Book Recommendation System.
    
    Attributes:
        books (pd.DataFrame): The dataset containing book metadata and emotions.
        vector_db (VectorDB): The vector database instance for semantic search.
    """
    def __init__(self) -> None:
        """Initialize the recommender by loading data and the vector database."""
        self.books = load_books_data()
        self.vector_db = VectorDB()
        
    def get_recommendations(
        self, 
        query: str, 
        category: str = "All", 
        tone: str = "All"
    ) -> List[Dict[str, Any]]:
        """
        Generate book recommendations based on query, category, and tone.
        Returns a list of dictionaries with book details.
        """
        try:
            if not query or not query.strip():
                return []

            logger.info(f"Processing request: query='{query}', category='{category}', tone='{tone}'")
            
            # 1. Semantic Search
            recs = self.vector_db.search(query, k=TOP_K_INITIAL)
            books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
            
            # 2. Filter by ISBN
            book_recs = self.books[self.books["isbn13"].isin(books_list)].head(TOP_K_INITIAL)
            
            # 3. Filter by Category
            if category and category != "All":
                book_recs = book_recs[book_recs["simple_categories"] == category].head(TOP_K_FINAL)
            else:
                book_recs = book_recs.head(TOP_K_FINAL)
                
            # 4. Sort by Tone
            if tone != "All":
                tone_map = {
                    "Happy": "joy",
                    "Surprising": "surprise",
                    "Angry": "anger",
                    "Suspenseful": "fear",
                    "Sad": "sadness"
                }
                if tone in tone_map:
                    book_recs = book_recs.sort_values(by=tone_map[tone], ascending=False)

            return self._format_results(book_recs)

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []

    def _format_results(self, book_recs: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Format the raw DataFrame results into a list of dictionaries for the API.
        
        Args:
            book_recs (pd.DataFrame): The filtered and sorted recommendations.
            
        Returns:
            List[Dict[str, Any]]: List of book objects with title, authors, etc.
        """
        results = []
        for _, row in book_recs.iterrows():
            description = row["description"]
            truncated_desc = " ".join(description.split()[:30]) + "..."
            
            authors = row["authors"].split(";")
            if len(authors) == 2:
                authors_str = f"{authors[0]} and {authors[1]}"
            elif len(authors) > 2:
                authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
            else:
                authors_str = row["authors"]
                
            results.append({
                "isbn": row["isbn13"],
                "title": row["title"],
                "authors": authors_str,
                "description": truncated_desc,
                "thumbnail": row["large_thumbnail"],
                "caption": f"{row['title']} by {authors_str}: {truncated_desc}"
            })
        return results

    def get_categories(self) -> List[str]:
        return ["All"] + sorted(self.books["simple_categories"].unique())

    def get_tones(self) -> List[str]:
        return ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
