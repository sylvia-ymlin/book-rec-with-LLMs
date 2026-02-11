import pandas as pd
import random

def generate_synthetic_data(num_samples: int = 100) -> pd.DataFrame:
    """
    Generates synthetic e-commerce product data.
    """
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Toys']
    adjectives = ['Premium', 'Budget', 'High-end', 'Durable', 'Stylish', 'Compact', 'Professional']
    products_map = {
        'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Smartwatch', 'Camera'],
        'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Dress'],
        'Home & Kitchen': ['Blender', 'Coffee Maker', 'Desk Lamp', 'Sofa', 'Curtains'],
        'Books': ['Novel', 'Textbook', 'Biography', 'Cookbook', 'Comic'],
        'Toys': ['Lego Set', 'Action Figure', 'Board Game', 'Puzzle', 'Doll']
    }
    
    data = []
    for i in range(num_samples):
        cat = random.choice(categories)
        prod = random.choice(products_map[cat])
        adj = random.choice(adjectives)
        
        title = f"{adj} {prod} {i+1}"
        price = round(random.uniform(10.0, 1000.0), 2)
        description = f"This is a {adj.lower()} {prod.lower()} perfect for your needs. It features high quality materials and modern design."
        features = f"Feature A, Feature B, {adj} Quality"
        
        data.append({
            'product_id': f"P{str(i).zfill(4)}",
            'title': title,
            'category': cat,
            'price': price,
            'description': description,
            'features': features,
            'review_text': f"Great {prod}! I loved the {adj.lower()} aspect."
        })
        
    return pd.DataFrame(data)

def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Loads data from a file or generates synthetic data if path is None.
    """
    if file_path:
        # Check extension and load accordingly
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
    else:
        print("No file path provided. Generating synthetic data...")
        return generate_synthetic_data()

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    df.to_csv("synthetic_products.csv", index=False)
    print("Saved synthetic_products.csv")
