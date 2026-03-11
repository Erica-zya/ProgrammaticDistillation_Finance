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


def load_processed_qids(path: Path):
    if not path.exists():
        return set()
    qids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "qid" in obj:
                    qids.add(obj["qid"])
            except Exception:
                pass
    return qids


@app.function(
    gpu="A100-80GB",
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=24 * 3600,
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

    success_path = out_dir / "recovered_successes.jsonl"
    failure_path = out_dir / "recovered_failures.jsonl"
    summary_path = out_dir / "recover_summary.json"

    processed_success_qids = load_processed_qids(success_path)
    processed_failure_qids = load_processed_qids(failure_path)
    processed_qids = processed_success_qids | processed_failure_qids

    with wrong_path.open("r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    data = [ex for ex in all_data if ex["qid"] not in processed_qids]

    print("num wrong samples total:", len(all_data))
    print("already processed:", len(processed_qids))
    print("remaining to process:", len(data))

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

    num_samples_per_q = 5
    max_new_tokens = 768
    temperature = 0.7
    top_p = 0.95
    save_every = 20

    new_successes = 0
    new_failures = 0

    success_f = success_path.open("a", encoding="utf-8")
    failure_f = failure_path.open("a", encoding="utf-8")

    try:
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
                    success_f.write(json.dumps(success_ex, ensure_ascii=False) + "\n")
                    success_f.flush()
                    new_successes += 1
                    break

            if not recovered:
                fail_ex = dict(ex)
                fail_ex["attempts"] = attempt_records
                failure_f.write(json.dumps(fail_ex, ensure_ascii=False) + "\n")
                failure_f.flush()
                new_failures += 1

            if i % save_every == 0:
                current_total_success = len(processed_success_qids) + new_successes
                current_total_failure = len(processed_failure_qids) + new_failures
                current_total_processed = current_total_success + current_total_failure

                summary = {
                    "num_candidates_total": len(all_data),
                    "already_processed_before_run": len(processed_qids),
                    "processed_in_this_run": i,
                    "num_successes_total": current_total_success,
                    "num_failures_total": current_total_failure,
                    "success_rate_over_processed": round(
                        current_total_success / max(current_total_processed, 1), 4
                    ),
                    "num_samples_per_q": num_samples_per_q,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "save_every": save_every,
                }
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                finance_vol.commit()
                print(
                    f"processed {i}/{len(data)} in this run | "
                    f"new_successes={new_successes} | new_failures={new_failures} | "
                    f"total_successes={current_total_success} | total_failures={current_total_failure}"
                )

        final_total_success = len(load_processed_qids(success_path))
        final_total_failure = len(load_processed_qids(failure_path))
        final_total_processed = final_total_success + final_total_failure

        summary = {
            "num_candidates_total": len(all_data),
            "already_processed_before_run": len(processed_qids),
            "processed_in_this_run": len(data),
            "num_successes_total": final_total_success,
            "num_failures_total": final_total_failure,
            "success_rate_over_processed": round(
                final_total_success / max(final_total_processed, 1), 4
            ),
            "num_samples_per_q": num_samples_per_q,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "save_every": save_every,
            "status": "finished",
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        finance_vol.commit()

    finally:
        success_f.close()
        failure_f.close()


@app.local_entrypoint()
def main():
    recover_teacher_wrong.remote()