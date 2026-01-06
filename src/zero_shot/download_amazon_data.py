#!/usr/bin/env python3
"""
Amazon Review Data (2023) Downloader
Based on McAuley-Lab/Amazon-Reviews-2023
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

class Amazon2023Downloader:
    def __init__(self, category='Books', base_dir='./amazon_data'):
        self.category = category
        self.processed_dir = os.path.join(base_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def run(self, sample_size=50000):
        print(f"Loading Amazon 2023: {self.category}")
        
        # Config names: raw_review_{Category}, raw_meta_{Category}
        # e.g. raw_review_Books, raw_meta_Books
        review_conf = f"raw_review_{self.category}"
        meta_conf = f"raw_meta_{self.category}"
        
        print(f"Downloading Reviews ({review_conf})...")
        try:
            # User provided example uses trust_remote_code=True
            reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", review_conf, split="full", trust_remote_code=True)
        except Exception as e:
            print(f"Error loading reviews: {e}")
            return
            
        # Sample if needed
        if sample_size and len(reviews) > sample_size:
            print(f"Sampling {sample_size} from {len(reviews)} reviews...")
            reviews = reviews.shuffle(seed=42).select(range(sample_size))
            
        print(f"Downloading Metadata ({meta_conf})...")
        try:
            meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", meta_conf, split="full", trust_remote_code=True)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return

        # Process Interactions
        print("Processing Interactions...")
        interaction_list = []
        item_ids = set()
        
        for r in tqdm(reviews):
            # 2023 fields: rating, title, text, user_id, timestamp, asin, parent_asin
            # Use parent_asin as item_id if available (better for grouping variants)
            item_id = r.get('parent_asin', r.get('asin'))
            if not item_id: continue
            
            interaction_list.append({
                'user_id': r['user_id'],
                'item_id': item_id,
                'rating': r['rating'],
                'interested': 'Yes' if r['rating'] >= 4 else 'No',
                'timestamp': r['timestamp']
            })
            item_ids.add(item_id)
            
        # Process Metadata
        print("Processing Metadata...")
        meta_list = []
        # Create a lookup for efficiency if meta is huge?
        # HF datasets are iterable. For Books, meta is huge.
        # We can iterate and filter.
        
        # Convert meta dataset to a iterable to avoid loading everything if possible?
        # Actually load_dataset("full") likely loads or maps it.
        # Let's just iterate and match item_id.
        
        # Optimization: Build a dict? Books meta is ~3M items.
        # If we have 50k reviews, we have maybe 20-30k items.
        # Building a 3M item dict might be slow/OOM.
        # But we can iterate meta once.
        
        count = 0
        for m in tqdm(meta):
            pid = m.get('parent_asin', m.get('asin'))
            if pid in item_ids:
                # Extract fields
                title = m.get('title', '')
                desc = m.get('description', [])
                if isinstance(desc, list): desc = " ".join(desc)
                feat = m.get('features', [])
                if isinstance(feat, list): feat = " ".join(feat)
                
                full_text = f"{title}. {desc} {feat}"[:1000]
                
                meta_list.append({
                    'item_id': pid,
                    'title': title,
                    'category': m.get('main_category', 'Books'),
                    'description': full_text,
                    'price': m.get('price', None)
                })
                item_ids.remove(pid) # Optimization: stop if all found? No, duplicates?
                # Actually parent_asin should be unique in meta? Hopefully.
        
        # Save
        i_df = pd.DataFrame(interaction_list)
        m_df = pd.DataFrame(meta_list)
        
        i_path = os.path.join(self.processed_dir, f"{self.category}_interactions.json")
        m_path = os.path.join(self.processed_dir, f"{self.category}_metadata.json")
        
        i_df.to_json(i_path, orient='records', lines=True)
        m_df.to_json(m_path, orient='records', lines=True)
        
        print(f"Success! Saved {len(i_df)} interactions and {len(m_df)} items.")

if __name__ == "__main__":
    # category "Books" or "All_Beauty"
    downloader = Amazon2023Downloader(category='Books')
    downloader.run(sample_size=50000)
