"""
Run the manual prompt (code_generation.PROMPT_TMPL) on the same 11 samples
in sample_of_each_type.json. Score with the same TaTQA metric as check_errors
so you can compare: optimized prompt (check_errors) vs manual prompt (this script).

Usage (from repo root):
  python prompt_optimization/run_manual_prompt_on_sample.py
"""
import json
import os
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation_metrics"))
from tatqa_metric import TaTQAEmAndF1

from dotenv import load_dotenv
load_dotenv()

from teacher_model.code_generation import PROMPT_TMPL, call_hf_chat
from teacher_model.run_generated_code import exec_with_timeout
from optimize_prompt import load_data, SAMPLE_PATH, normalize_scale


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set. Use .env or export HF_TOKEN.")
    url = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")
    model = os.getenv("HF_MODEL_72B", "Qwen/Qwen2.5-72B-Instruct")

    dataset = load_data(SAMPLE_PATH)
    metric = TaTQAEmAndF1()
    results = []

    print(f"\n{'='*20} MANUAL PROMPT (code_generation.PROMPT_TMPL) on sample_of_each_type.json {'='*20}\n")

    for i, example in enumerate(dataset):
        prompt = PROMPT_TMPL.format(
            table=example.table,
            paras=example.paragraphs,
            question=example.question,
            derivation=getattr(example, "derivation", ""),
        )
        code = call_hf_chat(url, token, model, prompt)
        time.sleep(1)

        if not (code and code.strip()):
            passed = False
            pred_ans = None
            pred_scale_norm = ""
            stdout_last = ""
            error = "empty code"
            status = "empty"
        else:
            stdout, error, status = exec_with_timeout(code)
            lines = [ln for ln in stdout.strip().split("\n") if ln.strip()]
            stdout_last = lines[-1] if lines else ""

            try:
                parsed = json.loads(stdout_last)
                pred_ans = parsed.get("ans", stdout_last)
                raw_scale = parsed.get("scale", "")
            except json.JSONDecodeError:
                pred_ans = stdout_last
                raw_scale = ""
            pred_scale_norm = normalize_scale(raw_scale)

            gold_scale_norm = normalize_scale(getattr(example, "scale", ""))
            ground_truth = {
                "answer_type": getattr(example, "answer_type", "arithmetic"),
                "scale": gold_scale_norm,
                "answer": getattr(example, "answer", None),
            }
            metric(ground_truth, pred_ans, pred_scale=pred_scale_norm)
            em, f1, scale_score, _ = metric.get_overall_metric(reset=True)
            passed = (em == 1.0 and scale_score == 1.0)

        results.append(passed)
        status_str = "✅ PASS" if passed else "❌ FAIL"
        print(f"Sample {i+1}: {example.question[:70]}...")
        print(f"  {status_str}")

        if not passed:
            print("  --- Debug ---")
            print(f"  Expected: {example.answer} (scale={getattr(example, 'scale', '')})")
            print(f"  Got:      {pred_ans} (scale={pred_scale_norm})")
            print(f"  Exec status: {status}  |  Exec error: {error or '(none)'}")
            if not (code and code.strip()):
                print("  API returned no code (empty or failed).")
            else:
                print(f"  Last line of stdout: {repr(stdout_last[:100])}")
                if not stdout_last and error:
                    print(f"  Code snippet (first 400 chars):\n  {code.strip()[:400]}")
            print()

    n = len(results)
    correct = sum(results)
    pct = 100.0 * correct / n if n else 0
    print(f"\n{'='*60}")
    print(f"MANUAL PROMPT (PROMPT_TMPL) on sample_of_each_type.json: {correct}/{n} correct ({pct:.0f}%)")
    print(f"{'='*60}\n")
    print("Compare with: python prompt_optimization/check_errors.py  (optimized prompt)")


if __name__ == "__main__":
    main()
