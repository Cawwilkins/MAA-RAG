from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_name = "meta-llama/Llama-3.2-1B-Instruct"
local_dir = r".\models\ai_models\llama-3.2-1b-instruct"

os.makedirs(local_dir, exist_ok=True)
print(f"Downloading {model_name}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model (CPU-friendly)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # safest for CPU
    low_cpu_mem_usage=True
)

# Save locally
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

print(f"✅ Saved {model_name} to {local_dir}")