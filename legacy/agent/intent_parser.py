import re
from typing import Dict, Optional

class IntentParser:
    def __init__(self):
        # In a real scenario, this would be an LLM-based parser
        pass
        
    def parse(self, query: str) -> Dict[str, Optional[str]]:
        """
        Parses the user query into structured slots.
        """
        query = query.lower()
        
        intent = {
            'category': None,
            'budget': None,
            'style': None,
            'original_query': query
        }
        
        # Rule-based Category Extraction
        categories = ['laptop', 'phone', 'smartphone', 'headphone', 'camera', 'jeans', 'shirt', 'dress', 'shoe', 'blender', 'coffee', 'lamp', 'sofa', 'desk', 'toy', 'lego', 'book', 'novel']
        for cat in categories:
            if cat in query:
                intent['category'] = cat
                break # Take the first match for now
        
        # Rule-based Budget Extraction
        # Look for "under $100", "cheap", "expensive", "budget"
        if "cheap" in query or "budget" in query:
            intent['budget'] = "low"
        elif "expensive" in query or "premium" in query:
            intent['budget'] = "high"
        
        match = re.search(r'under \$?(\d+)', query)
        if match:
            intent['budget'] = f"<{match.group(1)}"
            
        # Rule-based Style/Feature Extraction (naïve) 
        # Everything else that is an adjective could be style
        styles = ['gaming', 'professional', 'casual', 'formal', 'black', 'red', 'blue', 'wireless', 'bluetooth']
        found_styles = []
        for style in styles:
            if style in query:
                found_styles.append(style)
        
        if found_styles:
            intent['style'] = ", ".join(found_styles)
            
        return intent

if __name__ == "__main__":
    parser = IntentParser()
    queries = [
        "I want a cheap gaming laptop",
        "Looking for a blue dress under $50",
        "wireless headphones for travel"
    ]
    
    for q in queries:
        print(f"Query: {q}")
        print(f"Parsed: {parser.parse(q)}")
        print("-" * 20)
