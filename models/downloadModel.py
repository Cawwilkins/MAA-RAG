from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

model_name = "google/flan-t5-large"
local_dir = r".\models\flan-t5-large"

os.makedirs(local_dir, exist_ok=True)
print(f"Downloading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

print(f"✅ Saved {model_name} to {local_dir}")