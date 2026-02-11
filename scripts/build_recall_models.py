import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import logging
from src.recall.itemcf import ItemCF
from src.recall.usercf import UserCF
from src.recall.popularity import PopularityRecall

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading training data...")
    df = pd.read_csv('data/rec/train.csv')
    
    # 1. ItemCF
    logger.info("--- Training ItemCF ---")
    itemcf = ItemCF()
    if itemcf.load():
        logger.info("ItemCF model already exists, skipping training.")
    else:
        itemcf.fit(df)
    
    # 2. UserCF
    logger.info("--- Training UserCF ---")
    # For UserCF, using full data might be slow if many users/items.
    # The current implementation has hot-item pruning (limit=2000).
    # 1M records, 114k users. 
    usercf = UserCF()
    if usercf.load():
         # Force retrain if we optimized logic? No, load() returns True if exists.
         # But I just changed logic, so I want to RETRAIN UserCF.
         pass
    
    # Just force retrain UserCF for now since I optimized it
    usercf.fit(df)
    
    # 3. Popularity
    logger.info("--- Training Popularity ---")
    pop = PopularityRecall()
    pop.fit(df)
    
    logger.info("Recall models built and saved successfully!")

if __name__ == "__main__":
    main()
