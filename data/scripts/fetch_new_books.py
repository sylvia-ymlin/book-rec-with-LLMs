#!/usr/bin/env python3
"""
Incremental Book Update Script.

Fetches recently published books from Google Books API and adds them to the local database.
Can be run manually or scheduled via cron for periodic updates.

Usage:
    python scripts/data/fetch_new_books.py [--categories CATEGORIES] [--year YEAR] [--max MAX]

Examples:
    # Fetch new books from current year (default behavior)
    python scripts/data/fetch_new_books.py --categories "fiction" --max 50
    
    # Fetch new books across multiple categories
    python scripts/data/fetch_new_books.py --categories "fiction,mystery,science fiction"
    
    # Explicitly specify year filter
    python scripts/data/fetch_new_books.py --year 2026 --categories "thriller"
    
    # Dry run (show what would be added without actually adding)
    python scripts/data/fetch_new_books.py --dry-run --categories "thriller"
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger
from src.core.rag.web_search import search_new_books_by_category, search_google_books
from src.data.stores.metadata_store import metadata_store
from src.core.recommendation_orchestrator import RecommendationOrchestrator

logger = setup_logger(__name__)

# Default categories to search
DEFAULT_CATEGORIES = [
    "fiction",
    "mystery",
    "thriller",
    "science fiction",
    "fantasy",
    "romance",
    "biography",
    "history",
    "self-help",
    "business",
]


def fetch_trending_books(
    categories: list[str],
    year: Optional[int] = None,
    max_per_category: int = 20,
    dry_run: bool = False,
) -> dict:
    """
    Fetch recently published books from Google Books for given categories.
    
    Args:
        categories: List of book categories to search
        year: Filter by publication year (default: current year)
        max_per_category: Max books to fetch per category
        dry_run: If True, don't actually add books to database
    
    Returns:
        Dict with stats: {added: int, skipped: int, errors: int, books: list}
    """
    if year is None:
        year = datetime.now().year
    
    stats = {
        "added": 0,
        "skipped": 0,
        "errors": 0,
        "books": [],
    }
    
    recommender = None
    if not dry_run:
        recommender = RecommendationOrchestrator()
    
    for category in categories:
        logger.info(f"Fetching books for category: {category} (year >= {year})")
        
        try:
            books = search_new_books_by_category(
                category=category,
                year=year,
                max_results=max_per_category
            )
            
            logger.info(f"  Found {len(books)} books in '{category}'")
            
            for book in books:
                isbn = book.get("isbn13", "")
                if not isbn:
                    continue
                
                # Check if already exists
                if metadata_store.book_exists(isbn):
                    stats["skipped"] += 1
                    continue
                
                if dry_run:
                    logger.info(f"  [DRY RUN] Would add: {book.get('title', 'Unknown')} ({isbn})")
                    stats["books"].append(book)
                    stats["added"] += 1
                else:
                    result = recommender.add_new_book(
                        isbn=isbn,
                        title=book.get("title", ""),
                        author=book.get("authors", "Unknown"),
                        description=book.get("description", ""),
                        category=book.get("simple_categories", category),
                        thumbnail=book.get("thumbnail"),
                        published_date=book.get("publishedDate", ""),
                    )
                    
                    if result:
                        stats["added"] += 1
                        stats["books"].append(book)
                        logger.info(f"  Added: {book.get('title', 'Unknown')} ({isbn})")
                    else:
                        stats["errors"] += 1
                
                # Rate limiting: avoid hitting API limits
                time.sleep(0.1)
            
            # Pause between categories
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error fetching category '{category}': {e}")
            stats["errors"] += 1
    
    return stats


def fetch_by_query(
    queries: list[str],
    max_per_query: int = 20,
    dry_run: bool = False,
) -> dict:
    """
    Fetch books by specific search queries (e.g., "AI books 2024", "new thriller novels").
    
    Args:
        queries: List of search queries
        max_per_query: Max books per query
        dry_run: If True, don't actually add books
    
    Returns:
        Stats dict
    """
    stats = {
        "added": 0,
        "skipped": 0,
        "errors": 0,
        "books": [],
    }
    
    recommender = None
    if not dry_run:
        recommender = RecommendationOrchestrator()
    
    for query in queries:
        logger.info(f"Searching: {query}")
        
        try:
            books = search_google_books(query, max_results=max_per_query)
            logger.info(f"  Found {len(books)} results")
            
            for book in books:
                isbn = book.get("isbn13", "")
                if not isbn:
                    continue
                
                if metadata_store.book_exists(isbn):
                    stats["skipped"] += 1
                    continue
                
                if dry_run:
                    logger.info(f"  [DRY RUN] Would add: {book.get('title', 'Unknown')}")
                    stats["books"].append(book)
                    stats["added"] += 1
                else:
                    result = recommender.add_new_book(
                        isbn=isbn,
                        title=book.get("title", ""),
                        author=book.get("authors", "Unknown"),
                        description=book.get("description", ""),
                        category=book.get("simple_categories", "General"),
                        thumbnail=book.get("thumbnail"),
                        published_date=book.get("publishedDate", ""),
                    )
                    
                    if result:
                        stats["added"] += 1
                        stats["books"].append(book)
                    else:
                        stats["errors"] += 1
                
                time.sleep(0.1)
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error with query '{query}': {e}")
            stats["errors"] += 1
    
    return stats


def print_stats(stats: dict, dry_run: bool = False):
    """Print summary statistics."""
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}=== Fetch Complete ===")
    print(f"  Books added:   {stats['added']}")
    print(f"  Books skipped: {stats['skipped']} (already in database)")
    print(f"  Errors:        {stats['errors']}")
    
    if stats["books"] and dry_run:
        print(f"\nBooks that would be added:")
        for book in stats["books"][:10]:
            print(f"  - {book.get('title', 'Unknown')} by {book.get('authors', 'Unknown')}")
        if len(stats["books"]) > 10:
            print(f"  ... and {len(stats['books']) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch new books from Google Books API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories (default: all common categories)"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Comma-separated list of custom search queries"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter by publication year (default: current year)"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="Max books per category/query (default: 20)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be added without actually adding"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Parse categories
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    else:
        categories = DEFAULT_CATEGORIES
    
    # Parse queries
    queries = None
    if args.queries:
        queries = [q.strip() for q in args.queries.split(",")]
    
    print(f"Book Fetch Configuration:")
    print(f"  Categories: {categories if not queries else 'N/A (using queries)'}")
    print(f"  Queries: {queries or 'N/A (using categories)'}")
    print(f"  Year filter: >= {args.year or datetime.now().year}")
    print(f"  Max per item: {args.max}")
    print(f"  Dry run: {args.dry_run}")
    print()
    
    # Fetch books
    if queries:
        stats = fetch_by_query(
            queries=queries,
            max_per_query=args.max,
            dry_run=args.dry_run,
        )
    else:
        stats = fetch_trending_books(
            categories=categories,
            year=args.year,
            max_per_category=args.max,
            dry_run=args.dry_run,
        )
    
    print_stats(stats, args.dry_run)
    
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
