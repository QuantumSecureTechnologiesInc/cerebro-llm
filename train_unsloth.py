"""
Cerebro Unsloth Fine-Tuning Script
Run this locally on a GPU machine or in Colab.
"""
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import os

# Configuration
MODEL_NAME = "cerebro-tiny"  # or path to checkpoint
DATASET_PATH = "Manusagents/mega-distillation"  # or local JSONL path
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
MAX_STEPS = 200
OUTPUT_DIR = "cerebro_lora_output"

def main():
    print("Loading model...")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading dataset...")
    # Load dataset
    if os.path.exists(DATASET_PATH):
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    else:
        dataset = load_dataset(DATASET_PATH, split="train")
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Format for training
    def format_sample(sample):
        if "messages" in sample:
            return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}
        if "instruction" in sample and "response" in sample:
            return {"text": f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"}
        return {"text": str(sample.get("text", sample))}
    
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        warmup_steps=5,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        save_steps=50,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save
    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
