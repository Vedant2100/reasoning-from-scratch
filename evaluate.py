import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# Model paths
models = {
    "Base Model": "Qwen/Qwen2.5-0.5B",
    "Stage 1 (SFT)": "./coldstart_model",
    "Stage 2 (Dorado)": "./dorado_toy_model"
}

# Evaluation Prompts
eval_prompts = [
    "What is 3+4?",
    "Solve 1+2",
    "What is 10-3?",
    "Calculate 5*2",
    "Explain why 1+1=2"
]

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

def main():
    results = []
    
    for name, path in models.items():
        print(f"Evaluating {name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(path if "model" in path else "Qwen/Qwen2.5-0.5B")
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            for prompt in eval_prompts:
                response = generate_response(model, tokenizer, prompt)
                results.append({
                    "Model": name,
                    "Prompt": prompt,
                    "Response": response
                })
        except Exception as e:
            print(f"Could not load {name}: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        print("\n--- Evaluation Results ---")
        print(df.to_string())
        df.to_csv("evaluation_results.csv", index=False)
        print("\nResults saved to evaluation_results.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
