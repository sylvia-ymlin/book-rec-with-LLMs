#!/usr/bin/env python3
"""
Kaggle 'Amazon Books Reviews' Processor (Ratings Only Mode)
Adaptable to missing metadata file.
"""

import os
import pandas as pd
import json
import zipfile
from tqdm import tqdm

class KaggleBooksProcessor:
    def __init__(self, data_dir='amazon_data'):
        self.data_dir = data_dir
        self.output_dir = os.path.join(data_dir, 'processed')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.zip_file = os.path.join(data_dir, 'Books_rating.csv.zip')
        self.rating_file = os.path.join(data_dir, 'Books_rating.csv')
        self.meta_file = os.path.join(data_dir, 'books_data.csv') # Optional
        
    def check_and_unzip(self):
        if not os.path.exists(self.rating_file):
            if os.path.exists(self.zip_file):
                print(f"Unzipping {self.zip_file}...")
                with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
            else:
                print(f"❌ File not found: {self.rating_file} or {self.zip_file}")
                return False
        return True

    def run(self, sample_size=200000):
        print(f"Processing Data in {self.data_dir}...")
        
        if not self.check_and_unzip():
            return

        # 1. Load Ratings
        print("Loading Ratings (Books_rating.csv)...")
        # Columns: Id, Title, Price, User_id, profileName, review/helpfulness, review/score, review/time, review/summary, review/text
        # We use sampling for demo speed
        if sample_size:
             df = pd.read_csv(self.rating_file, nrows=sample_size)
        else:
             df = pd.read_csv(self.rating_file)
             
        print(f"Loaded {len(df)} records.")
        
        # 2. Extract Items & Interactions
        # Since we might lack books_data.csv, we rely on 'Title' in rating file.
        
        print("Extracting Metadata & Interactions...")
        interactions = []
        items_dict = {}
        
        # We iterate and build both
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                title = str(row['Title']).strip()
                if not title or title.lower() == 'nan': continue
                
                # Use Title as ID
                item_id = title 
                
                # Build Item Metadata (Simulated from Title if no meta file)
                if item_id not in items_dict:
                    price = row.get('Price', 'Unknown')
                    # We treat Title as 'Description' for basic Semantic Matching
                    full_desc = f"Title: {title}. Price: {price}."
                    
                    items_dict[item_id] = {
                        'item_id': item_id,
                        'title': title,
                        'category': 'Books', # Default
                        'description': full_desc,
                        'price': price
                    }
                
                interactions.append({
                    'user_id': str(row['User_id']),
                    'item_id': item_id,
                    'rating': float(row['review/score']),
                    'interested': 'Yes' if float(row['review/score']) >= 4.0 else 'No',
                    'timestamp': row.get('review/time', 0)
                })
            except:
                continue
                
        # 3. Save
        meta_out = pd.DataFrame(list(items_dict.values()))
        inter_out = pd.DataFrame(interactions)
        
        m_path = os.path.join(self.output_dir, 'kaggle_books_metadata.json')
        i_path = os.path.join(self.output_dir, 'kaggle_books_interactions.json')
        
        meta_out.to_json(m_path, orient='records', lines=True)
        inter_out.to_json(i_path, orient='records', lines=True)
        
        print(f"Done! Saved {len(meta_out)} items and {len(inter_out)} interactions.")
        print(f"  -> {m_path}")
        print(f"  -> {i_path}")

if __name__ == "__main__":
    p = KaggleBooksProcessor()
    # Process 200k rows for efficiency
    p.run(sample_size=200000)
