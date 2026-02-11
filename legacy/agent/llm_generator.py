from typing import List, Dict, Optional
import os

class LLMGenerator:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        Initialize LLM.
        Args:
            model_name: HuggingFace model name (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct').
                        If None, uses a Mock generator.
            device: 'cpu' or 'cuda'.
        """
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        
        if self.model_name and self.model_name != "mock":
            try:
                from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
                import torch
                
                print(f"Loading LLM: {model_name} on {device}...")
                # Note: In a real script, we would handle quantization (bitsandbytes) here
                # based on the device capabilities we discussed.
                dtype = torch.float16 if device == 'cuda' else torch.float32
                
                self.pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    torch_dtype=dtype,
                    device_map="auto" if device == 'cuda' else "cpu"
                )
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                print("Falling back to Mock Generator.")
                self.model_name = "mock"

    def generate_response(self, user_query: str, retrieved_items: List[Dict], history_str: str) -> str:
        """
        Generates a natural language response based on context.
        """
        # 1. Format retrieved items
        items_str = ""
        for i, item in enumerate(retrieved_items):
            items_str += f"{i+1}. {item['title']} (${item['price']}): {item['description']}\n"
            
        # 2. Construct Prompt (Simple Template)
        prompt = f"""You are a helpful shopping assistant.
        
Context History:
{history_str}

Retrieved Products related to the user's request:
{items_str}

User's Query: {user_query}

Instructions:
- Recommend the best products from the list above.
- Explain WHY they fit the user's request (budget, style, category).
- Be concise and friendly.

Response:"""

        if self.model_name == "mock" or self.model_name is None:
            return self._mock_generation(items_str)
        else:
            # Real LLM Generation
            try:
                outputs = self.pipeline(
                    prompt, 
                    max_new_tokens=200, 
                    do_sample=True, 
                    temperature=0.7,
                    truncation=True
                )
                generated_text = outputs[0]['generated_text']
                # Extract only the response part if the model echos the prompt (common in base pipelines)
                if "Response:" in generated_text:
                    return generated_text.split("Response:")[-1].strip()
                return generated_text
            except Exception as e:
                return f"[Error generating response: {e}]"

    def _mock_generation(self, items_str):
        """
        Fallback logic for testing without a GPU.
        """
        if not items_str:
            return "I couldn't find any products matching your specific criteria. Could you try different keywords?"
        
        return f"Based on your request, I found these great options:\n{items_str}\nI recommend checking the first one as it offers the best value!"
