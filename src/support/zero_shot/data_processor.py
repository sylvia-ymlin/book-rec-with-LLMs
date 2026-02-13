import pandas as pd

def process_data(input_file: str, output_file: str):
    """
    Converts ID-based features to semantic text prompts.
    
    Args:
        input_file: Path to raw interaction data.
        output_file: Path to save processed prompts.
    """
    print(f"Reading data from {input_file}...")
    # TODO: Load real data
    # df = pd.read_csv(input_file)
    
    # Dummy data
    data = {
        'item_id': [101, 102],
        'category': ['Electronics', 'Books'],
        'title': ['Wireless Headphones', 'Science Fiction Novel']
    }
    df = pd.DataFrame(data)
    
    print("Converting features to prompts...")
    df['prompt'] = df.apply(lambda x: f"Item: {x['title']} (Category: {x['category']})", axis=1)
    
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Preview:")
    print(df['prompt'].head())

if __name__ == "__main__":
    process_data("dummy_interactions.csv", "processed_prompts.csv")
