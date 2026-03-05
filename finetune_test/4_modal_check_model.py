import os
import modal

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers>=4.37.0",
    "sentencepiece",
    "accelerate",
    "huggingface_hub",
)

app = modal.App("finance-check-model", image=image)

hf_cache_vol = modal.Volume.from_name("hf-cache")

@app.function(
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=3600,
)
def check_model():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print("resolving local snapshot path from hf-cache...")
    local_path = snapshot_download(
        repo_id=model_name,
        cache_dir="/root/.cache/huggingface",
        local_files_only=True,
    )
    print("local snapshot path:", local_path)
    print("snapshot files:", os.listdir(local_path))

    print("loading tokenizer from local snapshot...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    print("loading model from local snapshot...")
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )

    print("model loaded.")
    print("first param device:", next(model.parameters()).device)

    prompt = "What is the capital of China? Be concise."
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    outputs = model.generate(**inputs, max_new_tokens=32)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== MODEL OUTPUT ===")
    print(text)

    hf_cache_vol.commit()

@app.local_entrypoint()
def main():
    check_model.remote()