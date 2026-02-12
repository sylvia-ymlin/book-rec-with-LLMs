"""
Metadata enrichment: fetches metadata, enriches, and filters by category.
Single responsibility: data completion for recommendation results.
"""
from typing import List, Optional

from src.data.stores.metadata_store import metadata_store
from src.core.models import BookResponseDict
from src.core.response_formatter import format_book_response
from src.utils import enrich_book_metadata
from src.config import TOP_K_FINAL


def enrich_and_format(
    isbn_list: List[str],
    category: str = "All",
    max_results: int = TOP_K_FINAL,
    source: str = "local",
    metadata_store_inst=None,
) -> List[BookResponseDict]:
    """
    Enrich ISBN list with metadata and format into API response dicts.

    - Fetches metadata from MetadataStore
    - Enriches with cover/author fallback (enrich_book_metadata)
    - Filters by category if specified
    - Returns formatted dicts up to max_results

    Args:
        isbn_list: List of ISBN strings
        category: Category filter ("All" = no filter)
        max_results: Max number of results to return
        source: Source label for response (local, content_based, etc.)

    Returns:
        List of formatted book dicts ready for API response
    """
    store = metadata_store_inst if metadata_store_inst is not None else metadata_store
    results: List[BookResponseDict] = []

    for isbn in isbn_list:
        meta = store.get_book_metadata(str(isbn))
        meta = enrich_book_metadata(meta, str(isbn))

        if not meta:
            continue

        if category and category != "All":
            if meta.get("simple_categories") != category:
                continue

        results.append(format_book_response(meta, str(isbn), source))

        if len(results) >= max_results:
            break

    return results
