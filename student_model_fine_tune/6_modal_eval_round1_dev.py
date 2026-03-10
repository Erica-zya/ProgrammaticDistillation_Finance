"""
Run round 1 LoRA model on dev set, download predictions, and evaluate (EM/F1).

Usage:
  modal run student_model_fine_tune/6_modal_eval_round1_dev.py

Does: (1) run model on dev, (2) download predictions, (3) run tatqa_eval.
"""
import json
import io
import contextlib
import multiprocessing as mp
import re
import subprocess
import sys
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

_adapter_local = REPO_ROOT / "student_model_fine_tune" / "lora_round1" / "adapter"
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers>=4.37.0",
        "sentencepiece",
        "accelerate",
        "peft",
        "huggingface_hub",
    )
    .add_local_file(
        str(REPO_ROOT / "dataset_filtered" / "tatqa_dataset_dev_filtered.json"),
        "/root/dataset_filtered/tatqa_dataset_dev_filtered.json",
    )
)
if _adapter_local.exists():
    image = image.add_local_dir(str(_adapter_local), "/root/adapter_round1_local")

app = modal.App("finance-eval-round1-dev", image=image)

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


def build_prompt(table_text: str, paras_text: str, question: str) -> str:
    return PROMPT_TMPL.format(table=table_text, paras=paras_text, question=question)


def _worker_exec(code, q):
    buf = io.StringIO()
    try:
        import builtins as real_builtins
        safe_env = {k: getattr(real_builtins, k) for k in dir(real_builtins)}
        import json as json_mod
        import math
        safe_env.update({"json": json_mod, "math": math})
        clean_code = code.replace("\xa0", " ")
        with contextlib.redirect_stdout(buf):
            exec(clean_code, safe_env)
        q.put({"stdout": buf.getvalue(), "error": ""})
    except Exception as e:
        q.put({"stdout": buf.getvalue(), "error": repr(e)})


def exec_with_timeout(code, timeout_s=5.0):
    q = mp.Queue()
    p = mp.Process(target=_worker_exec, args=(code, q), daemon=True)
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(1)
        return "", "timeout"
    res = q.get() if not q.empty() else {"stdout": "", "error": "Worker error"}
    return res["stdout"], res["error"] if res["error"] else None


def extract_code(text: str) -> str:
    """Extract Python code, handling optional markdown fences."""
    text = text.strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def parse_exec_output(stdout: str):
    """Parse ans and scale from last line of stdout (JSON)."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None, ""
    last_line = lines[-1]
    try:
        parsed = json.loads(last_line)
        return parsed.get("ans"), parsed.get("scale", "")
    except Exception:
        return last_line, ""


@app.function(
    gpu="A10G",
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=4 * 3600,
)
def eval_round1_on_dev():
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mp.set_start_method("spawn", force=True)

    dev_path = Path("/root/dataset_filtered/tatqa_dataset_dev_filtered.json")
    adapter_vol = Path("/root/finance-data/outputs/lora_round1_4800/adapter")
    adapter_local = Path("/root/adapter_round1_local")
    out_path = Path("/root/finance-data/outputs/round1_dev_predictions.json")

    if adapter_vol.exists():
        adapter_path = adapter_vol
    elif adapter_local.exists():
        adapter_path = adapter_local
        print("Using adapter from local (bundled in image); volume path not found.")
    else:
        raise FileNotFoundError(
            "Adapter not found. Either: (1) Run 5_modal_train_lora_round1.py to put it on the volume, "
            "or (2) ensure student_model_fine_tune/lora_round1/adapter/ exists locally."
        )

    data = json.loads(dev_path.read_text(encoding="utf-8"))

    # Flatten to (uid, table_text, paras_text, question)
    examples = []
    for ctx in data:
        table_rows = ctx.get("table", {}).get("table", [])
        table_text = "\n".join(" | ".join(map(str, row)) for row in table_rows)
        paras = sorted(ctx.get("paragraphs", []), key=lambda p: p.get("order", 0))
        paras_text = "\n".join(p.get("text", "") for p in paras)
        for qa in ctx.get("questions", []):
            examples.append({
                "uid": qa["uid"],
                "table_text": table_text,
                "paras_text": paras_text,
                "question": qa.get("question", ""),
            })

    print(f"Dev examples: {len(examples)}")

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
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    predictions = {}
    max_new_tokens = 768

    for i, ex in enumerate(examples):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(examples)}")

        prompt = build_prompt(
            ex["table_text"],
            ex["paras_text"],
            ex["question"],
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_code = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        code = extract_code(raw_code)

        stdout, err = exec_with_timeout(code, timeout_s=5.0)
        pred_ans, pred_scale = parse_exec_output(stdout) if stdout else (None, "")

        if pred_ans is not None:
            predictions[ex["uid"]] = [pred_ans, pred_scale or ""]
        else:
            predictions[ex["uid"]] = [None, ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    finance_vol.commit()
    print(f"Saved predictions to {out_path}")
    print(f"Coverage: {len([v for v in predictions.values() if v[0] is not None])}/{len(predictions)}")


@app.local_entrypoint()
def main():
    eval_round1_on_dev.remote()

    # Download predictions and run evaluation locally
    out_dir = REPO_ROOT / "student_model_fine_tune" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "round1_dev_predictions.json"
    gold_path = REPO_ROOT / "dataset_filtered" / "tatqa_dataset_dev_filtered.json"

    print("\n--- Downloading predictions ---")
    r = subprocess.run(
        ["modal", "volume", "get", "--force", "finance-data", "outputs/round1_dev_predictions.json", str(out_dir)],
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        print("Download failed. Run manually: modal volume get finance-data outputs/round1_dev_predictions.json student_model_fine_tune/outputs/")
        return

    if not pred_path.exists():
        print("Predictions file not found after download.")
        return

    print("\n--- Running evaluation ---")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "evaluation_metrics" / "tatqa_eval.py"),
            "--gold_path", str(gold_path),
            "--pred_path", str(pred_path),
        ],
        cwd=str(REPO_ROOT / "evaluation_metrics"),
        check=True,
    )
