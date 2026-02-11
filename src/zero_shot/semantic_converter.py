"""
P1: Semantic Modeling - Convert ID-based features to natural language prompts.
"""
import pandas as pd
from typing import List, Dict
import random

def generate_synthetic_interactions(num_users: int = 50, num_items: int = 100, num_interactions: int = 500) -> pd.DataFrame:
    """Generate synthetic user-item interaction data for cold-start simulation."""
    categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports']
    item_titles = {
        'Electronics': ['Wireless Earbuds', 'Smart Watch', 'Portable Charger', 'Bluetooth Speaker'],
        'Books': ['Science Fiction Novel', 'Biography', 'Cookbook', 'Self-Help Guide'],
        'Clothing': ['Running Shoes', 'Winter Jacket', 'Cotton T-Shirt', 'Denim Jeans'],
        'Home': ['Coffee Maker', 'Desk Lamp', 'Air Purifier', 'Robot Vacuum'],
        'Sports': ['Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Hiking Backpack']
    }
    
    # Generate items
    items = []
    for i in range(num_items):
        cat = random.choice(categories)
        title = random.choice(item_titles[cat])
        items.append({
            'item_id': f'I{str(i).zfill(4)}',
            'title': f'{title} #{i}',
            'category': cat,
            'price': round(random.uniform(10, 500), 2)
        })
    items_df = pd.DataFrame(items)
    
    # Generate interactions (user clicked/bought item)
    # Generate users with preferences
    users = []
    for i in range(num_users):
        users.append({
            'user_id': f'U{str(i).zfill(4)}',
            'preferred_category': random.choice(categories)
        })
    
    # Generate interactions (user clicked/bought item if category matches preference)
    interactions = []
    for _ in range(num_interactions):
        user = random.choice(users)
        # 50% chance to pick item from preferred category, 50% random
        if random.random() < 0.5:
            # Pick from preferred category
            candidate_items = items_df[items_df['category'] == user['preferred_category']]
            if not candidate_items.empty:
                item = candidate_items.sample(1).iloc[0]
            else:
                item = items_df.sample(1).iloc[0]
        else:
            # Pick random item
            item = items_df.sample(1).iloc[0]
            
        # Label logic: High chance of interest if category matches preference
        if item['category'] == user['preferred_category']:
            label = 1 if random.random() < 0.8 else 0  # 80% interest in preferred cat
        else:
            label = 0 if random.random() < 0.9 else 1  # 10% interest in other cats
            
        interactions.append({
            'user_id': user['user_id'],
            'item_id': item['item_id'],
            'label': label,
            'user_pref': user['preferred_category'] # Store for prompt context if needed
        })
    
    interactions_df = pd.DataFrame(interactions)
    return items_df, interactions_df

import os

def load_real_data(data_dir='amazon_data/processed'):
    """Load real processed Amazon data if available."""
    meta_path = os.path.join(data_dir, 'kaggle_books_metadata.json')
    inter_path = os.path.join(data_dir, 'kaggle_books_interactions.json')
    
    if not os.path.exists(meta_path) or not os.path.exists(inter_path):
        return None, None
        
    print(f"Loading real data from {data_dir}...")
    items_df = pd.read_json(meta_path, orient='records', lines=True)
    # Ensure item_id is string
    items_df['item_id'] = items_df['item_id'].astype(str)
    
    interactions_df = pd.read_json(inter_path, orient='records', lines=True)
    interactions_df['item_id'] = interactions_df['item_id'].astype(str)
    
    # Map 'interested' (Yes/No) to label (1/0)
    interactions_df['label'] = interactions_df['interested'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # SAMPLING FOR DEMO SPEED: Keep only 10k samples
    if len(interactions_df) > 10000:
        print(f"Sampling 10,000 interactions from {len(interactions_df)} for fast training...")
        interactions_df = interactions_df.sample(n=10000, random_state=42)
    
    # Add 'user_pref' column simulated from interactions or category
    # For Zero-shot simulation, if we don't have user profiles, we can infer preference from the distinct categories a user liked.
    # But since our kaggle fake interactions already implicitely used category, let's derive it.
    # Actually, let's merging item category back to interaction to simulate "User likes this category"
    if 'user_pref' not in interactions_df.columns:
        # Simple heuristic: The user's preference IS the category of the item they liked.
        # This is a bit leak-y but fine for Zero-shot "If user likes History, do they like this History book?"
         interactions_df = interactions_df.merge(items_df[['item_id', 'category']], on='item_id', how='left')
         interactions_df.rename(columns={'category': 'user_pref'}, inplace=True)
    
    return items_df, interactions_df

def convert_to_prompt(item: Dict, user_history: List[str] = None) -> str:
    """Convert item features to natural language prompt for LLM."""
    # Enhanced for Real Data with Description
    desc = item.get('description', '')
    # Truncate description to avoid exceeding token limit
    if len(desc) > 300:
        desc = desc[:300] + "..."
        
    prompt = f"Item: {item['title']}\nCategory: {item['category']}\nPrice: {item.get('price', 'N/A')}\nDescription: {desc}"
    
    if user_history:
        # Context: "User is interested in [Category]"
        prompt += f"\nUser's Context: Interested in {', '.join(user_history[:1])}"
    return prompt

def create_training_data(items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> List[Dict]:
    """Create training samples in instruction-tuning format."""
    training_data = []
    # items_map = items_df.set_index('item_id').to_dict('index') 
    # Optimization: items_df might have duplicates if ID not unique? 
    items_df = items_df.drop_duplicates(subset=['item_id'])
    items_map = items_df.set_index('item_id').to_dict('index')
    
    print("Generating prompts...")
    for _, row in interactions_df.iterrows():
        item_id = str(row['item_id'])
        item = items_map.get(item_id, {})
        if not item:
            continue
            
        instruction = "Based on the item description and user context, predict whether the user would be interested (Yes/No)."
        
        # In real data, user_pref might be the category of the positive sample
        user_pref = row.get('user_pref', item.get('category', 'Books'))
        input_text = convert_to_prompt(item, user_history=[str(user_pref)])
        
        output_text = "Yes" if row['label'] == 1 else "No"
        
        training_data.append({
            'instruction': instruction,
            'input': input_text,
            'output': output_text
        })
    
    return training_data

if __name__ == "__main__":
    # Try loading real data first
    # Fix path: script is in src/, data is in src/amazon_data/processed
    # When running from src/ dir:
    real_items, real_inters = load_real_data('amazon_data/processed')
    
    if real_items is not None:
        print("Using REAL Kaggle Data.")
        items_df, interactions_df = real_items, real_inters
    else:
        print("Real data not found. Generating SYNTHETIC data.")
        items_df, interactions_df = generate_synthetic_interactions()
        
    training_data = create_training_data(items_df, interactions_df)
    
    # Save as JSON for training
    import json
    # Save to current dir so trainer can find it easily
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Generated {len(training_data)} samples. Saved to training_data.json")
    print("Sample:", training_data[0] if training_data else "None")
