import pandas as pd
import numpy as np
from src.config import BOOKS_CSV, COVER_NOT_FOUND
from src.utils import setup_logger

logger = setup_logger(__name__)

def load_books_data() -> pd.DataFrame:
    """Load and preprocess the books dataset."""
    try:
        if not BOOKS_CSV.exists():
            raise FileNotFoundError(f"Books data file not found at {BOOKS_CSV}")
            
        logger.info(f"Loading books data from {BOOKS_CSV}")
        books = pd.read_csv(BOOKS_CSV)
        
        # Process thumbnails
        books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
        books["large_thumbnail"] = np.where(
            books["large_thumbnail"].isna(),
            str(COVER_NOT_FOUND),
            books["large_thumbnail"],
        )
        
        return books
    except Exception as e:
        logger.error(f"Error loading books data: {str(e)}")
        raise
