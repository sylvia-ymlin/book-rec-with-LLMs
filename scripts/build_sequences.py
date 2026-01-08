import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_sequences(data_dir='data/rec', max_len=50):
    """
    Convert user interactions into sequences for SASRec
    """
    logger.info("Building user sequences...")
    
    # Load all data to map ISBNs to Integers (SASRec needs int IDs)
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/val.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    full_df = pd.concat([train_df, val_df, test_df])
    
    # 1. Map ISBN to Index (1-based, 0 is padding)
    items = full_df['isbn'].unique()
    item_map = {isbn: i+1 for i, isbn in enumerate(items)}
    num_items = len(items)
    
    logger.info(f"Total items: {num_items}")
    
    # Save map
    with open(f'{data_dir}/item_map.pkl', 'wb') as f:
        pickle.dump(item_map, f)
        
    # 2. Group by User and Sort by Time
    # Note: Our data split script ALREADY sorted by time.
    # But let's be safe. We need original timestamps if possible.
    # 'train.csv' doesn't have timestamp column? let me check split_rec_data.
    # Ah, split_rec_data removed it. But rows are ordered.
    # Actually, we can just group by user_id and assume rows are chronological 
    # IF we process train -> val -> test order.
    
    # Let's reconstruct full history per user
    logger.info("Grouping user history...")
    
    # Optimization: processing via dictionary is faster than groupby on large df
    user_history = {} # user_id -> list of item_ids
    
    def process_df(df):
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            u = row['user_id']
            item = item_map[row['isbn']]
            if u not in user_history:
                user_history[u] = []
            user_history[u].append(item)
            
    # Process in chronological order: Train -> Val -> Test
    # Wait, split_rec_data puts last item in test, 2nd last in val.
    # So correct order is: Train rows + Val row + Test row.
    
    # BUT, train.csv has multiple rows per user. They are already sorted by time in split logic.
    process_df(train_df)
    process_df(val_df)
    
    # We leave Test item out of the input sequence! 
    # Test item is the target for evaluation.
    # For training SASRec, we use (seq[:-1]) -> predict (seq[1:]).
    
    # 3. Create Dataset
    # Output: 
    # train_seqs: Dict[user_id, list_of_ints]
    
    # Pad/Truncate
    final_seqs = {}
    
    for u, history in user_history.items():
        # Truncate to max_len
        seq = history[-max_len:]
        final_seqs[u] = seq
        
    logger.info(f"Processed {len(final_seqs)} users.")
    
    # Save sequences
    with open(f'{data_dir}/user_sequences.pkl', 'wb') as f:
        pickle.dump(final_seqs, f)
        
    logger.info("Sequence data saved.")
    
if __name__ == "__main__":
    build_sequences()
