import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from peft import LoraConfig
from datasets import Dataset
import numpy as np

# Configuration
model_name = "Qwen/Qwen2.5-0.5B"
sft_model_path = "./coldstart_model"
output_dir = "./dorado_toy_model"

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Stage 1 SFT model with value head for PPO
    # Note: If stage1 only saved adapters, we need to load base + adapters
    try:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            sft_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception:
        print("SFT model not found, falling back to base model for demo purposes.")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    ref_model = create_reference_model(model)

    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1.4e-5,
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=4,
        gradient_accumulation_steps=1,
    )

    # Toy reasoning dataset
    data = {
        "query": [
            "What is 3+4?",
            "Solve 1+2",
            "What is 10-3?",
            "Calculate 5*2"
        ],
        "target": ["7", "3", "7", "10"]
    }
    dataset = Dataset.from_dict(data)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    dataset = dataset.map(tokenize, batched=False)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # Reward functions
    def get_rewards(query_tensors, response_tensors, targets):
        rewards = []
        for query, response, target in zip(query_tensors, response_tensors, targets):
            resp_str = tokenizer.decode(response, skip_special_tokens=True).strip()
            
            # 1. Correctness Reward
            r_correct = 1.0 if resp_str == target.strip() else -1.0
            
            # 2. Preference Reward (Proxy: response length and formatting as placeholder for "style")
            # In a real Dorado setup, this would be a Reward Model score
            r_pref = 0.1 if len(resp_str) > 0 and "\n" not in resp_str else -0.1
            
            total_reward = torch.tensor(r_correct + 0.1 * r_pref)
            rewards.append(total_reward)
        return rewards

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 10,
    }

    print("Starting Stage 2: Dual-reward PPO...")
    for epoch in range(1): # Single epoch for toy demo
        for batch in ppo_trainer.dataloader:
            query_tensors = batch["input_ids"]
            
            # Generate responses
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            
            # Compute rewards
            rewards = get_rewards(query_tensors, response_tensors, batch["target"])
            
            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_pretrained(output_dir)
    print(f"Stage 2 complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
