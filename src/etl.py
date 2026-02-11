import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.config import DATA_DIR, COVER_NOT_FOUND
from src.utils import setup_logger

logger = setup_logger(__name__)

RAW_DATA_PATH = DATA_DIR / "raw" / "Books_rating.csv"
PROCESSED_DATA_PATH = DATA_DIR / "books_processed.csv"
REVIEW_HIGHLIGHTS_PATH = DATA_DIR / "review_highlights.txt"
DESCRIPTIONS_PATH = DATA_DIR / "books_descriptions.txt"

def load_books_data() -> pd.DataFrame:
    """
    Load and preprocess the Amazon books dataset.
    If processed file exists, load it. Otherwise, process raw data.
    """
    try:
        # Check if processed data exists
        if PROCESSED_DATA_PATH.exists():
            logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}")
            books = pd.read_csv(PROCESSED_DATA_PATH)
            # Ensure thumbnails are processed
            books["large_thumbnail"] = books["thumbnail"].fillna(str(COVER_NOT_FOUND))
            return books

        # Process raw data
        if not RAW_DATA_PATH.exists():
            logger.warning(f"Raw data file not found at {RAW_DATA_PATH}. Returning empty context.")
            # Return empty DataFrame with expected columns to avoid crashes
            return pd.DataFrame(columns=['isbn13', 'title', 'description', 'average_rating', 
                                       'authors', 'thumbnail', 'simple_categories',
                                       'joy', 'sadness', 'fear', 'anger', 'surprise',
                                       'large_thumbnail', 'tags', 'review_highlights'])

        logger.info(f"Processing raw data from {RAW_DATA_PATH}...")
        
        # Load Raw Data (Chunking if necessary, but 200MB fits in memory)
        # Columns: Id,Title,Price,User_id,profileName,review/helpfulness,
        #          review/score,review/time,review/summary,review/text
        df = pd.read_csv(RAW_DATA_PATH)
        
        # Data Cleaning
        df['Title'] = df['Title'].fillna("Unknown Title")
        df['review/text'] = df['review/text'].fillna("")
        df['review/summary'] = df['review/summary'].fillna("")
        
        # Aggregation Strategy: Group by Book ID (Id)
        # We need to synthesize a "Description" since the dataset is just reviews.
        # We'll take the top 3 longest reviews/summaries to represent the book.
        
        logger.info("Grouping reviews by book...")
        
        # Function to aggregate text
        def aggregate_text(series):
            # Sort by helpfullness or length? Length is a proxy for detail.
            # Simple approach: Concat first 3 reviews
            texts = series.head(3).tolist()
            return " ".join([str(t) for t in texts])[:1000] # Limit to 1000 chars

        grouped = df.groupby('Id').agg({
            'Title': 'first',
            'review/text': aggregate_text,
            'review/score': 'mean'
        }).reset_index()
        
        # Rename columns to match schema expected by Recommender
        grouped = grouped.rename(columns={
            'Id': 'isbn13',
            'Title': 'title',
            'review/text': 'description',
            'review/score': 'average_rating'
        })
        
        # Add missing columns with defaults (to be filled by future upgrades)
        grouped['authors'] = "Unknown"
        grouped['thumbnail'] = str(COVER_NOT_FOUND)
        grouped['simple_categories'] = "General"  # Default category
        
        # Add emotion columns (placeholders)
        for emotion in ['joy', 'sadness', 'fear', 'anger', 'surprise']:
            grouped[emotion] = 0.0

        # Save processed data
        logger.info(f"Saving processed data to {PROCESSED_DATA_PATH}")
        grouped.to_csv(PROCESSED_DATA_PATH, index=False)
        
        # Generate Descriptions TXT for VectorDB
        # Format: "ISBN Description"
        logger.info(f"Generating descriptions file at {DESCRIPTIONS_PATH}")
        with open(DESCRIPTIONS_PATH, 'w') as f:
            for _, row in grouped.iterrows():
                # Clean newlines from description
                clean_desc = str(row['description']).replace('\n', ' ')
                f.write(f"{row['isbn13']} {clean_desc}\n")
                
        # Final processing for return
        grouped["large_thumbnail"] = grouped["thumbnail"]
        
        return grouped

    except Exception as e:
        logger.error(f"Error process books data: {str(e)}")
        raise

if __name__ == "__main__":
    load_books_data()
