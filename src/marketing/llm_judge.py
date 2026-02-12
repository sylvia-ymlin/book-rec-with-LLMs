import os
import json
import random
from typing import Dict, Any

# You need to 'pip install openai'
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class MarketingJudge:
    def __init__(self, api_key: str = None, use_ollama: bool = True, ollama_model: str = "llama3"):
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        
        if self.use_ollama:
            # Connect to local Ollama instance (OpenAI-compatible API)
            try:
                self.client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama" # required but ignored
                )
                print(f"Judge Info: Connected to Local Ollama ({ollama_model}).")
            except Exception as e:
                self.client = None
                print(f"Judge Warning: Could not connect to Ollama: {e}")
        else:
            # Use Real OpenAI with explicitly provided API key only (no env fallback)
            self.api_key = api_key
            if self.api_key and OpenAI:
                self.client = OpenAI(api_key=self.api_key)
            else:
                self.client = None
                print("Judge Warning: No API key passed. Using Mock judge.")

    def evaluate(self, product_name: str, generated_copy: str, target_audience: str) -> Dict[str, Any]:
        """
        Uses an LLM (Ollama or GPT-4) to act as a Judge.
        """
        if not self.client:
            return self.mock_evaluate(product_name, generated_copy)

        prompt = f"""
        You are a Senior Marketing Editor. Evaluate the following analysis.
        
        Product: {product_name}
        Target Audience: {target_audience}
        Generated Copy: "{generated_copy}"
        
        Rate the copy on:
        1. Safety (Pass/Fail)
        2. Creativity (1-5)
        3. Alignment (1-5)
        
        Return JSON ONLY: {{ "safety": "Pass", "creativity_score": 4, "alignment_score": 5, "reasoning": "..." }}
        """

        try:
            model_id = self.ollama_model if self.use_ollama else "gpt-4"
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content
            # Cleanup for robust JSON parsing
            content = content.replace("```json", "").replace("```", "").strip()
            # Try to start from the first open brace if there is chatter
            if "{" in content:
                content = content[content.find("{"):content.rfind("}")+1]
                
            return json.loads(content)
        except Exception as e:
            print(f"Judge Error ({model_id}): {e}")
            return self.mock_evaluate(product_name, generated_copy)

    def mock_evaluate(self, product_name: str, generated_copy: str) -> Dict[str, Any]:
        """Simulates evaluation for demo purposes."""
        print("Judge Info: Falling back to Mock Judge.")
        # Simple heuristic: longer text = better score (just for mock)
        score = min(5, len(generated_copy) // 20 + 2)
        is_safe = "Pass" if "scam" not in generated_copy.lower() else "Fail"
        
        return {
            "safety": is_safe,
            "creativity_score": random.randint(3, 5),
            "alignment_score": score,
            "reasoning": "[MOCK] The copy mentions key features but could be more punchy."
        }

if __name__ == "__main__":
    # Test
    judge = MarketingJudge()
    
    test_product = "Space Pen"
    test_copy = "Discover the Space Pen! Write in zero gravity. Perfect for astronauts."
    audience = "Astronauts"
    
    print(f"Evaluating: {test_copy}")
    result = judge.evaluate(test_product, test_copy, audience)
    print(json.dumps(result, indent=2))
