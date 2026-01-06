"""
P2 & P3: LoRA Fine-tuning for Zero-shot Recommendation.
Optimized for RTX 3090/4090 (24GB VRAM).
"""
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from modelscope import snapshot_download

# ========== Configuration ==========
MODEL_NAME = snapshot_download("qwen/Qwen2-1.5B-Instruct")  # Load from ModelScope
OUTPUT_DIR = "./lora_output"
DATA_FILE = "training_data.json"

def load_model_and_tokenizer(model_name: str):
    """Load model with 4-bit quantization for memory efficiency."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def apply_lora(model):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Common for Qwen/Llama
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_dataset(data_file: str):
    """Load and format dataset for SFT."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Format as chat/instruction format
    formatted = []
    for item in data:
        text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        formatted.append({"text": text})
    
    return Dataset.from_list(formatted)

def train(model, tokenizer, dataset):
    """Run SFT training with LoRA."""
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,  # Quick iteration; increase for production
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

def main():
    print("=== Zero-shot Recommender LoRA Training ===")
    
    # Step 1: Generate data if not exists
    if not os.path.exists(DATA_FILE):
        print("Generating training data...")
        from semantic_converter import generate_synthetic_interactions, create_training_data
        items_df, interactions_df = generate_synthetic_interactions(num_interactions=1000)
        training_data = create_training_data(items_df, interactions_df)
        with open(DATA_FILE, 'w') as f:
            json.dump(training_data, f)
        print(f"Generated {len(training_data)} samples.")
    
    # Step 2: Load model
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Step 3: Apply LoRA
    print("Applying LoRA adapters...")
    model = apply_lora(model)
    
    # Step 4: Load dataset
    print("Loading dataset...")
    dataset = load_dataset(DATA_FILE)
    
    # Step 5: Train
    print("Starting training...")
    train(model, tokenizer, dataset)
    
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
