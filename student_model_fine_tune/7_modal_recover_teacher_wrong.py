import json
import io
import contextlib
import multiprocessing as mp
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

app = modal.App("finance-recover-teacher-wrong", image=image)

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


def normalize_answer(x):
    if isinstance(x, list):
        return sorted(str(v).strip() for v in x)
    if x is None:
        return None
    return str(x).strip()


def is_correct(pred_ans, pred_scale, gold_ans, gold_scale):
    return (
        normalize_answer(pred_ans) == normalize_answer(gold_ans)
        and str(pred_scale or "").strip() == str(gold_scale or "").strip()
    )


def _worker_exec(code, q):
    buf = io.StringIO()
    try:
        import builtins as real_builtins
        safe_env = {k: getattr(real_builtins, k) for k in dir(real_builtins)}

        import json, math
        safe_env.update({"json": json, "math": math})

        with contextlib.redirect_stdout(buf):
            exec(code, safe_env)

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
        return {"status": "timeout", "stdout": "", "error": "Timeout"}

    res = q.get() if not q.empty() else {"stdout": "", "error": "Worker error"}
    return {
        "status": "ok" if not res["error"] else "exec_error",
        "stdout": res["stdout"],
        "error": res["error"],
    }


@app.function(
    gpu="L4",
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=6 * 3600,
)
def recover_teacher_wrong():
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    wrong_path = Path("/root/finance-data/data/student_train_from_teacher_round2_wrong.jsonl")
    adapter_path = Path("/root/finance-data/outputs/lora_round1/adapter")
    out_dir = Path("/root/finance-data/outputs/recover_round1")
    out_dir.mkdir(parents=True, exist_ok=True)

    with wrong_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print("num wrong samples:", len(data))

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
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    successes = []
    failures = []

    num_samples_per_q = 5
    max_new_tokens = 768
    temperature = 0.7
    top_p = 0.95

    for i, ex in enumerate(data, 1):
        prompt = build_prompt(ex)
        gold_ans = ex.get("gold_answer")
        gold_scale = ex.get("gold_scale", "")

        recovered = False
        attempt_records = []

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        for attempt_idx in range(num_samples_per_q):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            code = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            exec_res = exec_with_timeout(code, timeout_s=5.0)

            pred_ans = None
            pred_scale = ""
            last_line = None

            if exec_res["stdout"]:
                lines = [ln.strip() for ln in exec_res["stdout"].splitlines() if ln.strip()]
                if lines:
                    last_line = lines[-1]
                    try:
                        parsed = json.loads(last_line)
                        pred_ans = parsed.get("ans")
                        pred_scale = parsed.get("scale", "")
                    except Exception:
                        pass

            record = {
                "attempt_idx": attempt_idx,
                "generated_code": code,
                "run_status": exec_res["status"],
                "stdout_last": last_line,
                "pred_answer": pred_ans,
                "pred_scale": pred_scale,
                "exec_error": exec_res["error"] or None,
            }
            attempt_records.append(record)

            if is_correct(pred_ans, pred_scale, gold_ans, gold_scale):
                recovered = True
                success_ex = {
                    "qid": ex["qid"],
                    "table": ex["table"],
                    "paragraphs": ex["paragraphs"],
                    "question": ex["question"],
                    "gold_answer": ex["gold_answer"],
                    "gold_scale": ex["gold_scale"],
                    "generated_code": code,
                    "source": "round1_recovered",
                    "recover_round": 1,
                    "recover_attempt": attempt_idx,
                    "num_attempts": num_samples_per_q,
                }
                successes.append(success_ex)
                break

        if not recovered:
            fail_ex = dict(ex)
            fail_ex["attempts"] = attempt_records
            failures.append(fail_ex)

        if i % 20 == 0:
            print(f"processed {i}/{len(data)} | successes={len(successes)} | failures={len(failures)}")

    with (out_dir / "recovered_successes.jsonl").open("w", encoding="utf-8") as f:
        for r in successes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (out_dir / "recovered_failures.jsonl").open("w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "num_candidates": len(data),
        "num_successes": len(successes),
        "num_failures": len(failures),
        "success_rate": round(len(successes) / max(len(data), 1), 4),
        "num_samples_per_q": num_samples_per_q,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    (out_dir / "recover_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    finance_vol.commit()


@app.local_entrypoint()
def main():
    recover_teacher_wrong.remote()