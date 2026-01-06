
"""
P3: Guardrails & Compliance for Marketing Content Engine.
Ensures generated content is safe, compliant, and on-brand.
"""
from typing import List, Dict, Optional
import re

class ContentGuardrail:
    def __init__(self):
        # Configuration for banned words (competitor mentions, sensitive topics)
        self.banned_words = [
            "competitor_x", "cheap quality", "fake", "scam", 
            "guaranteed to cure", "lose weight fast"
        ]
        
        # Tone configuration
        self.allowed_tones = ["professional", "enthusiastic", "friendly", "urgent"]
        
    def check_input_safety(self, prompt: str) -> bool:
        """Check if input prompt contains malicious or banned content."""
        prompt_lower = prompt.lower()
        for word in self.banned_words:
            if word in prompt_lower:
                print(f"Guardrail Alert: Input contains banned word '{word}'")
                return False
        return True
    
    def check_price_consistency(self, generated_text: str, true_price: float) -> bool:
        """
        Detects if the generated text contains a price that conflicts with the true price.
        Returns False if a conflicting price is found.
        """
        if true_price is None:
            return True
            
        # Regex to find prices like $99.99, $99
        # Finds '$' followed optionally by space, then digits, optionally dots and decimals
        price_patterns = re.findall(r'\$\s?(\d+(?:\.\d{1,2})?)', generated_text)
        
        for price_str in price_patterns:
            try:
                price_val = float(price_str)
                # Allow small tolerance (e.g. floating point issues)
                if abs(price_val - true_price) > 0.05:
                    print(f"Guardrail Alert: Price Hallucination! Found ${price_val}, expected ${true_price}")
                    return False
            except ValueError:
                continue
        return True

    def check_placeholders(self, generated_text: str) -> bool:
        """Check for leftover placeholders like [Name] or <INSERT DATE>."""
        # Matches content inside square brackets or angle brackets
        placeholders = re.findall(r'\[.*?\]|<.*?>', generated_text)
        if placeholders:
             print(f"Guardrail Alert: Placeholder tokens found: {placeholders}")
             return False
        return True

    def check_refusal(self, generated_text: str) -> bool:
         """Check if model refused to generate content."""
         refusal_phrases = [
             "as an ai language model",
             "i cannot generate",
             "i am unable to",
             "violate my safety guidelines",
             "inappropriate request"
         ]
         text_lower = generated_text.lower()
         for phrase in refusal_phrases:
             if phrase in text_lower:
                 print(f"Guardrail Alert: Model Refusal detected: '{phrase}'")
                 return False
         return True

    def check_output_safety(self, generated_text: str, true_price: Optional[float] = None) -> bool:
        """Check if generated output is compliant, safe, and accurate."""
        text_lower = generated_text.lower()
        
        # 1. Check for banned words
        for word in self.banned_words:
            if word in text_lower:
                print(f"Guardrail Alert: Output contains banned word '{word}'")
                return False
                
        # 2. Check minimal length
        if len(generated_text.strip()) < 10:
             print("Guardrail Alert: Output too short.")
             return False

        # 3. Check for Model Refusal (New)
        if not self.check_refusal(generated_text):
            return False

        # 4. Check for Placeholders (New)
        if not self.check_placeholders(generated_text):
            return False

        # 5. Check Price Consistency (New)
        if true_price is not None:
            if not self.check_price_consistency(generated_text, true_price):
                return False
             
        return True

    def validate_tone(self, text: str, target_tone: str) -> bool:
        """
        Simple tone check using keyword heuristics. 
        In production, this would use a classifier model.
        """
        # Placeholder logic
        return True

if __name__ == "__main__":
    # Test cases
    guard = ContentGuardrail()
    
    print("\n--- Basic Safety Checks ---")
    safe_output = "Wake up to perfection with our new BrewMaster 3000. Fresh coffee, every time."
    unsafe_output = "Don't buy from anyone else, they sell cheap quality garbage."
    print(f"Safe Output: {guard.check_output_safety(safe_output)}")
    print(f"Unsafe Output: {guard.check_output_safety(unsafe_output)}")
    
    print("\n--- Advanced Checks ---")
    # Price Hallucination
    hallucinated_price = "Get the new headphones for only $9.99!"
    print(f"Price Hallucination Check ($99.99 vs $9.99): {guard.check_output_safety(hallucinated_price, true_price=99.99)}")
    
    # Placeholder
    template_artifact = "Welcome to [INSERT COMPANY NAME]! We serve the best food."
    print(f"Placeholder Check: {guard.check_output_safety(template_artifact)}")
    
    # Refusal
    refusal_msg = "I cannot generate that content as it triggers my safety policy."
    print(f"Refusal Check: {guard.check_output_safety(refusal_msg)}")
