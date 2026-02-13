import pandas as pd
import json
import os
from typing import List, Dict

class DataPipeline:
    def __init__(self, raw_data_path: str, output_path: str):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        
    def load_data(self) -> pd.DataFrame:
        """Load raw product data from CSV."""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"File not found: {self.raw_data_path}")
        return pd.read_csv(self.raw_data_path)
    
    def construct_prompt(self, product: pd.Series) -> Dict:
        """Construct instruction-tuning sample for marketing copy generation."""
        # Template for marketing features
        name = product.get('name', 'Unknown Product')
        features = product.get('features', '')
        target_audience = product.get('target_audience', 'General')
        
        # Instruction
        instruction = f"Write a compelling marketing copy for a product targeting {target_audience}."
        
        # Input context
        input_text = f"Product: {name}\nKey Features: {features}"
        
        # Target Output (In a real scenario, this would come from a copywriter or existing high-quality dataset.
        # Here we simulate 'gold' output or use a placeholder for SFT if we had pairs.
        # For the purpose of this project, we might need synthetic generation if we don't have ground truth.
        # But assuming we have some 'marketing_copy' column in raw data:
        output_text = product.get('marketing_copy', '')
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }

    def run(self):
        """Execute the pipeline."""
        print(f"Loading data from {self.raw_data_path}...")
        df = self.load_data()
        
        print("Constructing prompts...")
        training_data = []
        for _, row in df.iterrows():
            sample = self.construct_prompt(row)
            if sample['output']: # Only keep samples with ground truth
                training_data.append(sample)
        
        print(f"Saving {len(training_data)} samples to {self.output_path}...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print("Pipeline complete.")

if __name__ == "__main__":
    # Example usage
    RAW_PATH = "../data/raw_products.csv"
    OUTPUT_PATH = "../data/training_data.json"
    
    # Create dummy data if not exists for testing
    if not os.path.exists(RAW_PATH):
        os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
        print("Creating dummy data for testing...")
        dummy_df = pd.DataFrame([
            {
                "name": "NoiseCancelling Headphones 700",
                "features": "Active Noise Cancellation, 20h Battery, Bluetooth 5.0",
                "target_audience": "Commuters",
                "marketing_copy": "Escape the chaos of the city. Immerse yourself in pure silence with the Headphones 700. Your perfect commute companion."
            },
            {
                "name": "Eco-Friendly Water Bottle",
                "features": "Stainless Steel, BPA Free, Keeps Cold for 24h",
                "target_audience": "Hikers",
                "marketing_copy": "Stay hydrated on every peak. Our durable, eco-friendly bottle keeps your water ice-cold while saving the planet."
            },
            {
                "name": "Smart Home Hub",
                "features": "Voice Control, Compatible with 500+ devices, Easy Setup",
                "target_audience": "Tech Enthusiasts",
                "marketing_copy": "Control your entire home with just your voice. The ultimate command center for the modern smart home."
            }
        ])
        dummy_df.to_csv(RAW_PATH, index=False)
    
    pipeline = DataPipeline(RAW_PATH, OUTPUT_PATH)
    pipeline.run()
