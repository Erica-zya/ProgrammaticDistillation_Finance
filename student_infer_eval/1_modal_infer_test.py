import json
from pathlib import Path
import modal

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers>=4.37.0",
    "sentencepiece",
    "accelerate",
    "peft",
    "huggingface_hub",
)

app = modal.App("finance-infer", image=image)

finance_vol = modal.Volume.from_name("finance-data")
hf_cache_vol = modal.Volume.from_name("hf-cache")

PROMPT_TMPL = """Write a Python program that answers the QUESTION.

Sources of truth:
- TABLE and PARAGRAPHS are the ONLY sources of facts and numbers.
- Determine the unit/scale ONLY from explicit cues in TABLE, QUESTION, or PARAGRAPHS.

Strict output rules:
- Output MUST be raw Python code only. Do NOT use Markdown fences.
- Do NOT print anything except the final answer.
- End with EXACTLY ONE print(...) statement (the very last line).

Answer requirements:
1) Compute the final answer and store it in a variable named ans.
2) ans MUST be either:
   - a string, OR
   - a list of strings (for questions with multiple valid answers).
3) If the QUESTION can have multiple answers (e.g., "which years", "list all", filtering conditions), set ans to a list of strings containing ALL matching answers.
4) Do not include duplicates in list answers.

Scale requirements (must match exactly one of these 5):
- The JSON field "scale" MUST be exactly one of: "", "thousand", "million", "billion", "percent".

Final output format:
- Create a dictionary named out with exactly these keys:
  out = {{"ans": ..., "scale": ...}}
- Print EXACTLY one line of valid JSON.
- The LAST line of the program MUST be:
  print(json.dumps(out, ensure_ascii=False))

TABLE:
{table}

PARAGRAPHS:
{paras}

QUESTION:
{question}
"""



def build_prompt(example: dict) -> str:
    return PROMPT_TMPL.format(
        table=example.get("table", ""),
        paras=example.get("paragraphs", ""),
        question=example.get("question", ""),
    )


@app.function(
    gpu="A100-80GB",
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=6 * 3600,
)
def smoke_test():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "/root/finance-data/outputs/lora_round1_4800/adapter"

    local_model_path = snapshot_download(
        repo_id=base_model_name,
        cache_dir="/root/.cache/huggingface",
        local_files_only=True,
    )
    print("local_model_path:", local_model_path)
    print("adapter_path:", adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True,
    )
    model.eval()

    example = {
        "table": "Year | Revenue\n2023 | 10\n2024 | 12",
        "paragraphs": "The company's revenue increased from 10 to 12.",
        "question": "What was the revenue in 2024?",
    }
    prompt = build_prompt(example)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    num_return_sequences = 3
    do_sample = num_return_sequences > 1

    gen_kwargs = {
        "max_new_tokens": 256,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }


    if do_sample:
        gen_kwargs.update({
            "temperature": 0.5,
            "top_p": 0.9,
        })

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )

    print("\n===== GENERATED TEXTS =====\n")
    for i, out in enumerate(outputs):
        generated_ids = out[input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"--- candidate {i+1} ---")
        print(generated_text)
        print()
    print("===== END =====\n")


@app.local_entrypoint()
def main():
    smoke_test.remote()