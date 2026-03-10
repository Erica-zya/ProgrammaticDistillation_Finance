import json
from pathlib import Path
import modal

###
# input file: student_train_from_teacher.jsonl, paragraph, table are processed.
###

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers>=4.37.0",
    "sentencepiece",
    "accelerate",
    "peft",
    "huggingface_hub",
)

app = modal.App("finance-generate-code-a", image=image)

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
    gpu="RTX-PRO-6000",
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=12 * 3600,
)
def generate_code_a(
    #limit: int = 0,
    num_return_sequences:int,
    adapter_path:str,
    in_path:str,
    out_path:str,
):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "Qwen/Qwen2.5-7B-Instruct"

    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print("base_model_name:", base_model_name)
    print("adapter_path:", adapter_path)
    print("in_path:", in_path)
    print("out_path:", out_path)
    print("num_return_sequences:", num_return_sequences)

    # load model
    local_model_path = snapshot_download(
        repo_id=base_model_name,
        cache_dir="/root/.cache/huggingface",
        local_files_only=True,
    )

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
    ###
    data = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    done = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(str(json.loads(line)["qid"]))

    written = 0
    do_sample = num_return_sequences > 1

    with out_path.open("a", encoding="utf-8") as f:
        for ex in data:
            #if limit and written >= limit:
             #   print(f"[DONE] Reached limit of {limit} samples.")
              #  finance_vol.commit()
               # return

            qid = str(ex.get("qid", "unknown"))
            if qid in done:
                continue

            prompt = build_prompt(ex)

            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]

                gen_kwargs = {
                    "max_new_tokens": 512,
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

                generated_codes = []
                for out in outputs:
                    generated_ids = out[input_len:]
                    generated_text = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    ).strip()
                    generated_codes.append(generated_text)

                rec = {
                    "qid": qid,
                    "table": ex.get("table", ""),
                    "paragraphs": ex.get("paragraphs", ""),
                    "question": ex.get("question", ""),
                    "gold_answer": ex.get("gold_answer"),
                    "gold_scale": ex.get("gold_scale", ""),
                    "generated_codes": generated_codes,
                }

            except Exception as e:
                rec = {
                    "qid": qid,
                    "table": ex.get("table", ""),
                    "paragraphs": ex.get("paragraphs", ""),
                    "question": ex.get("question", ""),
                    "gold_answer": ex.get("gold_answer"),
                    "gold_scale": ex.get("gold_scale", ""),
                    "generated_codes": [],
                    "status": "generation_error",
                    "error": str(e),
                }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

            if written % 10 == 0:
                print(f"Processed {written} samples...")

    finance_vol.commit()
    print(f"[DONE] Wrote {written} samples to {out_path}")


@app.local_entrypoint()
def main():
    model="lora_round1_no_max_seq_mlp_exp2"
    generate_code_a.remote(
        #limit=5,
        num_return_sequences=1,
        adapter_path="/root/finance-data/outputs/"+model+"/adapter",
        in_path="/root/finance-data/data/student_train_from_teacher_round2_wrong.jsonl",
        out_path="/root/finance-data/outputs/"+model+"/outputs/student_generate_code_round1_train_wrong_single.jsonl",
    )