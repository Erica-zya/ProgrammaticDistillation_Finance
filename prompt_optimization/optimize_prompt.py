import os
import json
from pathlib import Path
from datetime import datetime
from tatqa_metric import TaTQAEmAndF1

from dotenv import load_dotenv
import dspy
from dspy.teleprompt import MIPROv2

from teacher_model.run_generated_code import exec_with_timeout, to_float_maybe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = Path(__file__).resolve().parent / "sample_of_each_type.json"
DATA_DIR = PROJECT_ROOT / "dataset_filtered"
LOG_PATH = Path(__file__).resolve().parent / "opt_results.jsonl"

def log_jsonl(obj, path: Path = LOG_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set.")

lm = dspy.LM("huggingface/Qwen/Qwen2.5-72B-Instruct", api_key=HF_TOKEN)
dspy.configure(lm=lm)

#  define the signature (initial prompt) of the program
class FinancialCodegen(dspy.Signature):
    """
    Write a Python program to answer financial questions.

    Output requirements:
    - pred_scale must be one of: "", percent, thousand, million, billion
    - program must be valid Python code
    - The final print() MUST output a raw number only (no units, no commas, no %).
    - If pred_scale == "percent", the printed number should be the raw percent value
      (e.g., print 17.7), and the evaluator will apply the scale.
    """
    table = dspy.InputField(desc="Financial table in markdown/text format")
    paragraphs = dspy.InputField(desc="Supporting text context")
    question = dspy.InputField(desc="The question to answer")
    scale = dspy.InputField(desc="Gold scale label (may help); one of: '', percent, thousand, million, billion")

    pred_scale = dspy.OutputField(desc='Predicted scale: one of "", percent, thousand, million, billion')
    program = dspy.OutputField(desc="Python code that prints the numeric answer (raw number only)")

# evaluate the code (placeholder for now)

def normalize_scale(s):
    if s is None:
        return ""
    s = str(s).strip()

    # model sometimes outputs "" literally (with quotes)
    if s in ['""', "''", '"\"\""', "'\"\"'"]:
        return ""

    # strip wrapping quotes if present
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ["'", '"']):
        inner = s[1:-1].strip()
        if inner == "":
            return ""
        s = inner

    s = s.lower().strip()
    allowed = {"", "percent", "thousand", "million", "billion"}
    return s if s in allowed else ""



def validate_code_metric_v0(gold, pred, trace=None):
    stdout, error, status = exec_with_timeout(pred.program)

    lines = [ln for ln in stdout.strip().split('\n') if ln.strip()]
    stdout_last = lines[-1] if lines else ""

    pred_val = to_float_maybe(stdout_last)
    gold_val = to_float_maybe(gold.answer)

    pred_scale_norm = normalize_scale(getattr(pred, "pred_scale", ""))
    gold_scale = normalize_scale(getattr(gold, "scale", ""))

    passed = False
    if pred_val is not None and gold_val is not None:
        # 1) scale must match (when gold_scale is provided)
        scale_ok = (pred_scale_norm == gold_scale) if gold_scale != "" else (pred_scale_norm == "")

        # 2) numeric must match under the scale convention
        numeric_ok = False
        if gold_scale == "percent":
            # enforce the convention: print raw percent value (e.g., 17.7)
            numeric_ok = abs(pred_val - gold_val) < 1e-4
        else:
            numeric_ok = abs(pred_val - gold_val) < 1e-4

        passed = scale_ok and numeric_ok

    log_jsonl({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "qid": getattr(gold, "qid", None),
        "question": getattr(gold, "question", None),
        "gold_scale": getattr(gold, "scale", None),
        "pred_scale": pred_scale_norm,
        "gold_answer": gold.answer,
        "program": getattr(pred, "program", None),
        "status": status,
        "stdout_last": stdout_last,
        "pred_answer": pred_val,
        "exec_error": error,
        "passed": passed,
    })

    return passed
# load the data

tatqa_metric = TaTQAEmAndF1()

def validate_code_metric(gold, pred, trace=None):
    stdout, error, status = exec_with_timeout(pred.program)
    lines = [ln for ln in stdout.strip().split('\n') if ln.strip()]
    stdout_last = lines[-1] if lines else ""

    pred_val = to_float_maybe(stdout_last)

    pred_scale_norm = normalize_scale(getattr(pred, "pred_scale", ""))
    gold_scale_norm = normalize_scale(getattr(gold, "scale", ""))

    ground_truth = {
        "answer_type": "arithmetic",
        "scale": gold_scale_norm,
        "answer": getattr(gold, "answer", None),
    }

    tatqa_metric(ground_truth, stdout_last, pred_scale=pred_scale_norm)
    em, f1, scale_score, op_score = tatqa_metric.get_overall_metric(reset=True)

    passed = (em == 1.0 and scale_score == 1.0)

    log_jsonl({
        "qid": getattr(gold, "qid", None),
        "gold_scale": gold_scale_norm,
        "pred_scale": pred_scale_norm,
        "gold_answer": getattr(gold, "answer", None),
        "pred_answer": pred_val,
        "stdout_last": stdout_last,
        "status": status,
        "exec_error": error,
        "passed": passed,
        "em": em,
        "f1": f1,
        "scale_score": scale_score,
    })
    return passed



TRAIN_PATH = DATA_DIR / "tatqa_dataset_train_filtered.json"







def load_data(sample_json: Path, limit: int | None = None):
    """
    Build a small training set from sample_of_each_type.json plus the filtered dataset.

    sample_of_each_type.json structure (per key):
      {
        "source_split": "tatqa_dataset_train_filtered.json",
        "table_uid": "...",
        "paragraph_uids": [...],
        "question": { ... TAT-QA question object ... }
      }
    """
    spec_dict = json.loads(sample_json.read_text(encoding="utf-8"))

    # Load train filtered once and index by table uid.
    data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    ctx_by_uid = {
        ctx.get("table", {}).get("uid"): ctx for ctx in data
    }

    dataset: list[dspy.Example] = []

    for spec in spec_dict.values():
        if limit is not None and len(dataset) >= limit:
            break

        table_uid = spec["table_uid"]
        ctx = ctx_by_uid.get(table_uid)
        if ctx is None:
            continue

        table_rows = ctx.get("table", {}).get("table", [])
        table_text = "\n".join(" | ".join(map(str, row)) for row in table_rows)

        para_uid_set = set(spec.get("paragraph_uids", []))
        paras = [
            p.get("text", "")
            for p in ctx.get("paragraphs", [])
            if not para_uid_set or p.get("uid") in para_uid_set
        ]
        paragraphs_text = "\n".join(paras)

        q_obj = spec["question"]
        scale_val = q_obj.get("scale", "")

        example = dspy.Example(
            qid=q_obj.get("uid"),
            table=table_text,
            paragraphs=paragraphs_text,
            question=q_obj.get("question", ""),
            scale=scale_val,
            answer=q_obj.get("answer"),
        ).with_inputs("table", "paragraphs", "question", "scale")

        dataset.append(example)

    return dataset

if __name__ == '__main__':
    trainset = load_data(SAMPLE_PATH)

    # initialize MIPROv2 (let 'auto' choose search size)
    teleprompter = MIPROv2(metric=validate_code_metric, auto="light")

    # run the "Prompt Battle"
    optimized_program = teleprompter.compile(
        dspy.Predict(FinancialCodegen), 
        trainset=trainset,
        max_bootstrapped_demos=3, # Pick 3 best examples to put in prompt
        max_labeled_demos=2       # Pick 2 gold examples
    )

    optimized_program.save("prompt_optimization/best_financial_prompt.json")

    if hasattr(teleprompter, 'trials_df'):
        print("\n--- Optimization Results ---")
        # Displaying the top trials by score
        df = teleprompter.trials_df.sort_values(by='score', ascending=False)
        print(df[['instruction', 'score']].head(10))
        