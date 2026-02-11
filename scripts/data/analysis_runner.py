
import pandas as pd
import numpy as np
import re
import os

def analyze():
    print("Loading data...")
    # Load data
    try:
        train_df = pd.read_csv('data/rec/train.csv')
        books_df = pd.read_csv('data/books_processed.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Train rows: {len(train_df)}")
    print(f"Books rows: {len(books_df)}")

    # 1. Sparsity & Distributions
    n_users = train_df['user_id'].nunique()
    n_items = train_df['isbn'].nunique()
    sparsity = 1 - (len(train_df) / (n_users * n_items))
    
    print("\n--- Distribution Stats ---")
    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Sparsity: {sparsity:.6f}")
    
    user_counts = train_df['user_id'].value_counts()
    item_counts = train_df['isbn'].value_counts()
    
    print(f"User Ratings: Min={user_counts.min()}, Max={user_counts.max()}, Median={user_counts.median()}, Mean={user_counts.mean():.2f}")
    print(f"Item Ratings: Min={item_counts.min()}, Max={item_counts.max()}, Median={item_counts.median()}, Mean={item_counts.mean():.2f}")
    
    # Head and Tail check
    top_1_percent_users = user_counts.quantile(0.99)
    top_1_percent_items = item_counts.quantile(0.99)
    print(f"Top 1% User Threshold: {top_1_percent_users:.0f} ratings")
    print(f"Top 1% Item Threshold: {top_1_percent_items:.0f} ratings")

    # 2. Text Quality
    print("\n--- Text Quality ---")
    books_df['desc_len'] = books_df['description'].fillna("").apply(len)
    print(f"Description Length: Mean={books_df['desc_len'].mean():.0f}, Median={books_df['desc_len'].median():.0f}")
    print(f"Missing Descriptions: {sum(books_df['desc_len'] == 0)} ({sum(books_df['desc_len'] == 0)/len(books_df):.2%})")
    
    def has_html(text):
        if not isinstance(text, str): return False
        return bool(re.search(r'<[^>]+>', text))
    
    html_count = books_df['description'].apply(has_html).sum()
    print(f"Rows with HTML artifacts: {html_count}")

    # 3. Temporal
    print("\n--- Temporal Analysis ---")
    if 'timestamp' in train_df.columns:
        # Convert to datetime if numeric
        try:
            # Assuming timestamp is unix
            dates = pd.to_datetime(train_df['timestamp'], unit='s', errors='coerce')
            print(f"Date Range: {dates.min()} to {dates.max()}")
            
            # Interactions per year
            print("Ratings per Year:")
            print(dates.dt.year.value_counts().sort_index())
        except:
            print("Could not parse timestamps")

if __name__ == "__main__":
    analyze()
