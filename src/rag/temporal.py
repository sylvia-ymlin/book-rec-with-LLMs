from typing import Dict, Any, List
from datetime import datetime
import math

from src.infra.utils import setup_logger


logger = setup_logger(__name__)


class TemporalRanker:
    """
    Applies Time Decay to search results.
    Boosts newer documents based on 'publishedDate'.
    """

    def __init__(self):
        self.current_year = datetime.now().year

    def parse_year(self, date_str: Any) -> int:
        """Robustly extract year from various date formats."""
        if not date_str:
            return 0
        try:
            s = str(date_str).strip()
            # Handle "2005-01-01" or "1999"
            if len(s) >= 4 and s[:4].isdigit():
                return int(s[:4])
        except (ValueError, IndexError):
            pass
        return 0

    def apply_decay(
        self,
        docs: List[Any],
        pub_year_map: Dict[str, int],
        boost_factor: float = 0.25,
    ) -> List[Any]:
        """
        Boost scores of newer books.
        New Score = Old Score * (1 + boost_factor * recency_weight)
        Recency Weight = 1 / log(Age + 2)  (Soft decay)
        """
        boosted_docs = []

        for doc in docs:
            # 1. robust ID extraction (as per recommender)
            isbn = None
            if doc.metadata and "isbn" in doc.metadata and doc.metadata["isbn"]:
                isbn = str(doc.metadata["isbn"])
            elif (
                doc.metadata
                and "isbn13" in doc.metadata
                and doc.metadata["isbn13"]
            ):
                isbn = str(doc.metadata["isbn13"])
            elif "ISBN:" in doc.page_content:
                try:
                    isbn = (
                        doc.page_content.split("ISBN:")[1]
                        .strip()
                        .split()[0]
                    )
                except (IndexError, AttributeError):
                    pass
            if not isbn and doc.page_content:
                isbn = doc.page_content.strip().split()[0]

            # 2. Get Year
            pub_year = pub_year_map.get(isbn, 0)

            # 3. Calculate Boost
            if pub_year > 1900:  # Valid year
                age = max(0, self.current_year - pub_year)
                # Exponential-ish decay
                recency_weight = 1.0 / math.log(age + 2.718)  # log(e)=1 at age 0
            else:
                recency_weight = 0.0

            # 4. Update Score (if exists) or create it
            if doc.metadata and "relevance_score" in doc.metadata:
                original_score = doc.metadata["relevance_score"]
                # Additive boost for logits, capped
                boost = 2.0 * recency_weight if pub_year > 1900 else 0
                doc.metadata["relevance_score"] = original_score + boost
                doc.metadata["year"] = pub_year  # Debug info

            boosted_docs.append(doc)

        # Resort
        boosted_docs.sort(
            key=lambda x: x.metadata.get("relevance_score", 0)
            if x.metadata
            else 0,
            reverse=True,
        )
        return boosted_docs


temporal_ranker = TemporalRanker()

__all__ = ["TemporalRanker", "temporal_ranker"]

