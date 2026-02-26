"""
Run the optimized prompt (e.g. best_financial_prompt.json or best_financial_prompt_n100_seed42.json)
on the first N samples from tatqa_dataset_train_filtered.json and report EM / F1 using the same
TaTQA evaluation as in the DSPy metric.

Usage (from repo root):
  python prompt_optimization/eval_optimized_on_train500.py
  python prompt_optimization/eval_optimized_on_train500.py --limit 100
  python prompt_optimization/eval_optimized_on_train500.py --prompt best_financial_prompt_n100_seed42.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test_metric"))
sys.path.insert(0, str(PROJECT_ROOT / "prompt_optimization"))
from tatqa_metric import TaTQAEmAndF1

from dotenv import load_dotenv
import dspy
from teacher_model.run_generated_code import exec_with_timeout

from optimize_prompt import FinancialCodegen, normalize_scale, DATA_DIR

load_dotenv()
TRAIN_PATH = DATA_DIR / "tatqa_dataset_train_filtered.json"
PROMPT_DIR = Path(__file__).resolve().parent


def iter_train_questions(limit: int):
    """Yield (table_text, paragraphs_text, q) for the first `limit` questions in train filtered."""
    data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    n = 0
    for ctx in data:
        if n >= limit:
            break
        table_rows = ctx.get("table", {}).get("table", [])
        table_text = "\n".join(" | ".join(map(str, row)) for row in table_rows)
        paragraphs_text = "\n".join(p.get("text", "") for p in ctx.get("paragraphs", []))
        for q in ctx.get("questions", []):
            if n >= limit:
                break
            yield table_text, paragraphs_text, q
            n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="Number of train questions to evaluate")
    ap.add_argument("--prompt", type=str, default="best_financial_prompt.json",
                    help="Prompt JSON filename (e.g. best_financial_prompt_n100_seed42.json) or path")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set. Use .env or export HF_TOKEN.")

    prompt_path = Path(args.prompt)
    if not prompt_path.is_absolute():
        prompt_path = PROMPT_DIR / prompt_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    dspy.configure(lm=dspy.LM("huggingface/Qwen/Qwen2.5-72B-Instruct", api_key=token))
    compiled = dspy.Predict(FinancialCodegen)
    compiled.load(prompt_path)

    metric = TaTQAEmAndF1()
    samples = list(iter_train_questions(args.limit))
    n_total = len(samples)

    print(f"Evaluating optimized prompt ({prompt_path.name}) on first {n_total} train samples (from {TRAIN_PATH.name})...")
    print()

    for i, (table_text, paragraphs_text, q) in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i + 1}/{n_total} ...")
        kwargs = dict(table=table_text, paragraphs=paragraphs_text, question=q.get("question", ""))
        if "derivation" in q:
            kwargs["derivation"] = q.get("derivation", "")
        try:
            pred = compiled(**kwargs)
            code = getattr(pred, "program", "") or ""
        except Exception:
            code = ""
        if not (code and code.strip()):
            pred_ans = None
            pred_scale_norm = ""
        else:
            stdout, _err, _status = exec_with_timeout(code)
            lines = [ln for ln in stdout.strip().split("\n") if ln.strip()]
            last = lines[-1] if lines else ""
            try:
                parsed = json.loads(last)
                pred_ans = parsed.get("ans", last)
                raw_scale = parsed.get("scale", "")
            except json.JSONDecodeError:
                pred_ans = last
                raw_scale = ""
            pred_scale_norm = normalize_scale(raw_scale)
        gold_scale_norm = normalize_scale(q.get("scale", ""))
        ground_truth = {
            "answer_type": q.get("answer_type", "arithmetic"),
            "scale": gold_scale_norm,
            "answer": q.get("answer"),
        }
        metric(ground_truth, pred_ans if pred_ans is not None else [], pred_scale=pred_scale_norm)

    em, f1, scale_em, _ = metric.get_overall_metric(reset=False)
    print()
    print("=" * 60)
    print(f"Optimized prompt ({prompt_path.name}) on first {n_total} train samples")
    print("=" * 60)
    print(f"  EM:     {em * 100:.2f}%")
    print(f"  F1:     {f1 * 100:.2f}%")
    print(f"  Scale:  {scale_em * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
