from intent_parser import IntentParser
from rag_retriever import ProductRetriever
from dialogue_manager import DialogueManager
from llm_generator import LLMGenerator
import os

class ShoppingAgent:
    def __init__(self, index_path: str, metadata_path: str, llm_model: str = None):
        self.parser = IntentParser()
        self.retriever = ProductRetriever(index_path, metadata_path)
        self.dialogue_manager = DialogueManager()
        self.llm = LLMGenerator(model_name=llm_model) # Defaults to mock
        
    def process_query(self, query: str):
        print(f"\nUser: {query}")
        
        # 1. Parse Intent
        intent = self.parser.parse(query)
        # print(f"[Debug] Intent: {intent}")
        
        # 2. Enrich Query (incorporating history could happen here)
        search_query = query
        if intent['category']:
            search_query += f" {intent['category']}"
        
        # 3. Retrieve
        results = self.retriever.search(search_query, k=3)
        
        # 4. Generate Response using LLM + History
        history_str = self.dialogue_manager.get_context_string()
        response = self.llm.generate_response(query, results, history_str)
        
        # 5. Update Memory
        self.dialogue_manager.add_turn(query, response)
            
        print("[Agent]:")
        print(response)
        return response

    def reset(self):
        self.dialogue_manager.clear_history()

if __name__ == "__main__":
    if not os.path.exists("data/product_index.faiss"):
        print("Index not found. Please run rag_indexer.py first.")
    else:
        # Pass "mock" to force CPU-friendly mock generation, 
        # or pass a model name like "gpt2" (small) if you have 'transformers' installed to test pipeline.
        agent = ShoppingAgent("data/product_index.faiss", "data/product_metadata.pkl", llm_model="mock")
        
        print("--- Turn 1 ---")
        agent.process_query("I need a gaming laptop under $1000")
        
        print("\n--- Turn 2 ---")
        agent.process_query("Do you have anything cheaper?")
