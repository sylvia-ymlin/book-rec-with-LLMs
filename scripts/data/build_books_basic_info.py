import csv
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run(
    books_path: Path = Path("data/books_data.csv"),
    ratings_path: Path = Path("data/Books_rating.csv"),
    output_path: Path = Path("data/books_basic_info.csv"),
) -> None:
    """Build books basic info from raw data. Callable from Pipeline."""
    books_data = pd.read_csv(
        str(books_path),
        engine="python",
        quotechar='"',
        escapechar='\\',
        on_bad_lines='skip',
    )
    ratings = pd.read_csv(
        str(ratings_path),
        engine="python",
        quotechar='"',
        escapechar='\\',
        on_bad_lines='skip',
    )

    books_cols = ["Title", "description", "authors", "image", "publisher", "publishedDate", "categories"]
    books_data = books_data[books_cols]
    ratings = ratings[["Title", "Id", "review/score"]].drop_duplicates(subset=["Title"])
    merged = books_data.merge(ratings, on="Title", how="left")
    merged = merged.rename(columns={
        "Id": "isbn10", "Title": "title", "authors": "authors", "description": "description",
        "image": "image", "publisher": "publisher", "publishedDate": "publishedDate",
        "categories": "categories", "review/score": "average_rating"
    })
    merged["isbn13"] = None

    merged.to_csv(str(output_path), index=False, quoting=csv.QUOTE_ALL, quotechar='"', escapechar='\\')
    logger.info("Saved %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
