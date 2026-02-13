
"""
P2: SFT Trainer for Marketing Content Engine.
Fine-tune Qwen2-7B-Instruct using QLoRA.
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
# Use 7B model for higher quality generation
MODEL_ID = "qwen/Qwen2-7B-Instruct"  
OUTPUT_DIR = "./sft_output"
DATA_FILE = "../data/training_data.json"

def load_model_and_tokenizer():
    """Load 7B model with 4-bit quantization."""
    print(f"Downloading/Loading model: {MODEL_ID}...")
    model_dir = snapshot_download(MODEL_ID)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing to save VRAM for 7B model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def apply_lora(model):
    """Apply LoRA adapters."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_dataset(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)
        
    formatted = []
    for item in data:
        # Chat format for Qwen
        text = f"<|im_start|>user\n{item['instruction']}\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        formatted.append({"text": text})
        
    return Dataset.from_list(formatted)

def train(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2, # Smaller batch size for 7B
        gradient_accumulation_steps=8, # Increase accumulation
        learning_rate=1e-4,
        warmup_steps=10,
        logging_steps=1,
        save_steps=20,
        bf16=True, # Critical for Ampere+
        optim="paged_adamw_8bit",
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer  # Updated API
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Data file {DATA_FILE} not found! Run pipeline_builder.py first.")
        return

    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    dataset = load_dataset(DATA_FILE)
    
    print("Starting training on Qwen2-7B...")
    train(model, tokenizer, dataset)
    print("Training Complete.")

if __name__ == "__main__":
    main()
