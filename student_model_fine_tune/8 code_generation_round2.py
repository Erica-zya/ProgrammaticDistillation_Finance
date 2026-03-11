import json
import time
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

app = modal.App("finance-round2-codegen-eval", image=image)

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

SPLIT_MAP = {
    "dev": "tatqa_dataset_dev_filtered.json",
    "test": "tatqa_dataset_test_filtered.json",
}


def build_prompt(table_text: str, paras_text: str, question: str) -> str:
    return PROMPT_TMPL.format(
        table=table_text,
        paras=paras_text,
        question=question,
    )


@app.function(
    gpu="A100-80GB",
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=24 * 3600,
)
def generate_split(split="dev", limit=0):
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if split not in SPLIT_MAP:
        raise ValueError(f"Unsupported split: {split}")

    in_path = Path("/root/finance-data/data") / SPLIT_MAP[split]
    out_dir = Path("/root/finance-data/outputs/round2_codegen_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"student_codegen_round2_{split}.jsonl"

    print("input path:", in_path)
    print("output path:", out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    done = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(str(json.loads(line)["qid"]))

    data = json.loads(in_path.read_text(encoding="utf-8"))

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    local_model_path = snapshot_download(
        repo_id=model_name,
        cache_dir="/root/.cache/huggingface",
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
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

    adapter_path = "/root/finance-data/outputs/lora_round2/adapter"
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for doc in data:
            doc_id = doc.get("table", {}).get("uid", "unknown")
            table_text = "\n".join(" | ".join(map(str, r)) for r in doc.get("table", {}).get("table", []))
            paras_text = "\n".join(p.get("text", "") for p in doc.get("paragraphs", []))

            for qi, q in enumerate(doc.get("questions", [])):
                if limit and written >= limit:
                    finance_vol.commit()
                    print(f"[DONE] Reached limit={limit}")
                    return

                qid = str(q.get("uid", f"{doc_id}_{qi}"))
                if qid in done:
                    continue

                prompt = build_prompt(
                    table_text=table_text,
                    paras_text=paras_text,
                    question=q.get("question", ""),
                )

                t0 = time.time()
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        temperature=0.0,
                        max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                code = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                rec = {
                    "qid": qid,
                    "split": split,
                    "doc_id": doc_id,
                    "question": q.get("question", ""),
                    "gold_answer": q.get("answer"),
                    "gold_scale": q.get("scale", ""),
                    "generated_code": code,
                    "latency_sec": round(time.time() - t0, 3),
                    "status": "ok",
                    "student_model": "Qwen/Qwen2.5-7B-Instruct + lora_round2_4800",
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

                if written % 20 == 0:
                    f.flush()
                    finance_vol.commit()
                    print(f"Processed {written} new samples for split={split}...")

    finance_vol.commit()
    print(f"Finished split={split}, wrote {written} new records.")


@app.local_entrypoint()
def main(split: str = "dev", limit: int = 0):
    generate_split.remote(split, limit)