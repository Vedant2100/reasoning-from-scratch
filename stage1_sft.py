import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Using a very small model for "small GPU" compatibility
model_name = "Qwen/Qwen2.5-0.5B"
output_dir = "./coldstart_model"

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optional 4-bit quantization if bitsandbytes is available
    # For JupyterHub/Small GPUs, we use float16 or bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Apply LoRA for memory efficiency
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Use a small subset of self-instruct
    dataset = load_dataset("yizhongw/self_instruct", split="train[:500]")

    def tokenize_fn(ex):
        prompt = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nResponse: {ex['output']}"
        return tokenizer(prompt, truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting Stage 1: Cold-start SFT...")
    trainer.train()
    
    # Save the adapter
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Stage 1 complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
