
"""
P4: Verification for Marketing Content Engine.
Loads the fine-tuned model and verifies output against guardrails.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download
from guardrails import ContentGuardrail

# Config
BASE_MODEL_ID = "qwen/Qwen2-7B-Instruct"
LORA_PATH = "./sft_output"

def load_model():
    print("Loading base model...")
    model_dir = snapshot_download(BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("Loading LoRA adapters...")
    try:
        model = PeftModel.from_pretrained(model, LORA_PATH)
    except Exception as e:
        print(f"Warning: Could not load LoRA adapters: {e}")
        print("Running with base model only.")
        
    model.eval()
    return model, tokenizer

def generate_copy(model, tokenizer, features: str, audience: str):
    prompt = f"<|im_start|>user\nWrite a compelling marketing copy for a product targeting {audience}.\nProduct: Test Product\nKey Features: {features}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant response
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    return response

def main():
    guard = ContentGuardrail()
    model, tokenizer = load_model()
    
    test_cases = [
        {"features": "Organic, Fair Trade, Dark Roast", "audience": "Coffee Lovers"},
        {"features": "Cheap quality, fake leather", "audience": "Budget shoppers (Edge case)"} # Should trigger guardrail or be handled
    ]
    
    print("\n=== Verification Start ===")
    for case in test_cases:
        print(f"\nGenerating for: {case['features']} -> {case['audience']}")
        copy = generate_copy(model, tokenizer, case['features'], case['audience'])
        print(f"Generated Copy: {copy}")
        
        # Guardrail checks
        is_safe = guard.check_output_safety(copy)
        print(f"Guardrail Check: {'PASSED' if is_safe else 'FAILED'}")

if __name__ == "__main__":
    main()
