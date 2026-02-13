"""
Data Freshness Monitor for RAG.

Provides insights into the freshness of the local book database and helps
decide when to fall back to web search.
"""
from datetime import datetime
from typing import Optional

from src.data.stores.metadata_store import metadata_store
from src.infra.utils import setup_logger


logger = setup_logger(__name__)


class FreshnessMonitor:
    """
    Monitor data freshness and provide staleness detection.

    Usage:
        monitor = FreshnessMonitor()
        stats = monitor.get_data_stats()
        if monitor.is_stale_for_query("latest 2025 books"):
            # Trigger web search fallback
    """

    # Years considered "recent" for freshness calculations
    RECENT_YEARS_THRESHOLD = 2

    def __init__(self):
        self._cache = {}
        self._cache_timestamp = None
        self._cache_ttl_seconds = 300  # 5 minutes

    def _is_cache_valid(self) -> bool:
        """Check if cached stats are still valid."""
        if not self._cache or not self._cache_timestamp:
            return False
        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < self._cache_ttl_seconds

    def get_data_stats(self, force_refresh: bool = False) -> dict:
        """
        Get comprehensive statistics about data freshness.

        Returns:
            Dict with:
                - total_books: Total number of books in database
                - newest_year: Year of most recently published book
                - oldest_year: Year of oldest book
                - books_by_year: Dict mapping year -> count
                - recent_books_count: Books published in last N years
                - data_cutoff_year: Effective "knowledge cutoff" year
                - freshness_score: 0-100 score indicating data freshness
        """
        if not force_refresh and self._is_cache_valid():
            return self._cache

        stats = {
            "total_books": 0,
            "newest_year": None,
            "oldest_year": None,
            "books_by_year": {},
            "recent_books_count": 0,
            "data_cutoff_year": None,
            "freshness_score": 0,
            "last_checked": datetime.now().isoformat(),
        }

        try:
            stats["total_books"] = metadata_store.get_book_count()
            stats["books_by_year"] = (
                metadata_store.get_books_by_year_distribution()
            )

            if stats["books_by_year"]:
                years = sorted(stats["books_by_year"].keys())
                stats["newest_year"] = max(years)
                stats["oldest_year"] = min(years)
                stats["data_cutoff_year"] = stats["newest_year"]

                # Count recent books (last N years)
                current_year = datetime.now().year
                recent_threshold = current_year - self.RECENT_YEARS_THRESHOLD
                stats["recent_books_count"] = sum(
                    count
                    for year, count in stats["books_by_year"].items()
                    if year >= recent_threshold
                )

                # Calculate freshness score (0-100)
                # Based on: newest year relative to current year
                years_behind = current_year - (
                    stats["newest_year"] or current_year
                )
                stats["freshness_score"] = max(
                    0, 100 - (years_behind * 25)
                )

            self._cache = stats
            self._cache_timestamp = datetime.now()

        except Exception as e:
            logger.error("FreshnessMonitor.get_data_stats failed: %s", e)

        return stats

    def is_stale(self, target_year: Optional[int] = None) -> bool:
        """
        Check if local data is too old for a given target year.

        Args:
            target_year: Year the user is asking about (default: current year)

        Returns:
            True if data is stale and web fallback should be triggered
        """
        if target_year is None:
            target_year = datetime.now().year

        stats = self.get_data_stats()
        newest_year = stats.get("newest_year")

        if newest_year is None:
            return True  # No data at all

        # Stale if target year is newer than our newest data
        return target_year > newest_year

    def is_stale_for_query(self, query: str) -> bool:
        """
        Analyze a query and determine if data is stale for it.

        Args:
            query: User's search query

        Returns:
            True if web fallback should be triggered
        """
        from src.rag.web_search import extract_year_from_query

        target_year = extract_year_from_query(query)

        if target_year is None:
            # No year requirement - check freshness score
            stats = self.get_data_stats()
            # Trigger fallback if data is more than 2 years old
            return stats.get("freshness_score", 100) < 50

        return self.is_stale(target_year)

    def get_coverage_for_year(self, year: int) -> dict:
        """
        Get coverage statistics for a specific year.

        Returns:
            Dict with: count, percentage of total, is_well_covered
        """
        stats = self.get_data_stats()
        year_count = stats["books_by_year"].get(year, 0)
        total = stats["total_books"] or 1

        return {
            "year": year,
            "count": year_count,
            "percentage": round(year_count / total * 100, 2),
            "is_well_covered": year_count >= 100,  # Arbitrary threshold
        }

    def recommend_update_categories(self) -> list[str]:
        """
        Recommend categories that should be updated.

        Returns:
            List of category names that need fresh data
        """
        # This would require category-level year tracking
        # For now, return common categories that benefit from freshness
        return [
            "fiction",
            "thriller",
            "science fiction",
            "fantasy",
            "mystery",
            "self-help",
            "business",
        ]

    def get_summary(self) -> str:
        """
        Get a human-readable summary of data freshness.

        Returns:
            Formatted string describing data freshness status
        """
        stats = self.get_data_stats()

        lines = [
            "Data Freshness Report",
            "=" * 40,
            f"Total books: {stats['total_books']:,}",
            f"Newest book year: {stats['newest_year'] or 'Unknown'}",
            f"Data cutoff: {stats['data_cutoff_year'] or 'Unknown'}",
            f"Recent books (last {self.RECENT_YEARS_THRESHOLD} years): {stats['recent_books_count']:,}",
            f"Freshness score: {stats['freshness_score']}/100",
        ]

        current_year = datetime.now().year
        if stats["newest_year"] and stats["newest_year"] < current_year:
            years_behind = current_year - stats["newest_year"]
            lines.append("")
            lines.append(
                f"WARNING: Data is {years_behind} year(s) behind current year."
            )
            lines.append(
                f"Consider running: python data/scripts/fetch_new_books.py --year {current_year}"
            )

        return "\n".join(lines)


# Global instance
freshness_monitor = FreshnessMonitor()


def is_data_fresh_enough(query: str) -> bool:
    """
    Quick check if local data is fresh enough for a query.

    Args:
        query: User's search query

    Returns:
        True if local data is sufficient, False if web fallback recommended
    """
    return not freshness_monitor.is_stale_for_query(query)


__all__ = ["FreshnessMonitor", "freshness_monitor", "is_data_fresh_enough"]

