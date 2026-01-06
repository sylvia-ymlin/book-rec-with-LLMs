"""
P4: Evaluation - Test zero-shot and fine-tuned model performance.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download
import json

def load_finetuned_model(base_model: str, lora_path: str):
    """Load base model + LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, item_info: str) -> str:
    """Run inference on a single item."""
    prompt = f"### Instruction:\nBased on the following item information, predict whether the user would be interested (Yes/No).\n\n### Input:\n{item_info}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response

def evaluate(model, tokenizer, test_data: list) -> dict:
    """Evaluate model on test set."""
    correct = 0
    total = len(test_data)
    
    for sample in test_data:
        pred = predict(model, tokenizer, sample['input'])
        expected = sample['output']
        if expected.lower() in pred.lower():
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}

if __name__ == "__main__":
    import sys
    
    BASE_MODEL = snapshot_download("qwen/Qwen2-1.5B-Instruct")
    LORA_PATH = "./lora_output"
    
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model(BASE_MODEL, LORA_PATH)
    
    # Load test data (use last 100 samples from training data as pseudo-test)
    with open("training_data.json", 'r') as f:
        all_data = json.load(f)
    test_data = all_data[-100:]
    
    print(f"Evaluating on {len(test_data)} samples...")
    results = evaluate(model, tokenizer, test_data)
    
    print(f"Results: Accuracy = {results['accuracy']*100:.1f}% ({results['correct']}/{results['total']})")
