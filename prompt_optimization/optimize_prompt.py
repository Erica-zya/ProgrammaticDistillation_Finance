import os
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "test_metric"))
from tatqa_metric import TaTQAEmAndF1

from dotenv import load_dotenv
import dspy
from dspy.teleprompt import MIPROv2

from teacher_model.run_generated_code import exec_with_timeout, to_float_maybe
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
#  Matches teacher PROMPT_TMPL: program must print exactly one JSON line {"ans": ..., "scale": ...}
class FinancialCodegen(dspy.Signature):
    """
    Write a Python program that answers the QUESTION using TABLE and PARAGRAPHS.
    
    Instruction:
    1. Compute the final answer and store it in a variable named 'ans'.
    2. Multi-Span Rule: If the QUESTION asks for multiple items (e.g., 'which years', 'list all'), 'ans' MUST be a list of strings. Otherwise, 'ans' is a single string.
    3. Infer scale from context; 'scale' MUST be one of: "", "percent", "thousand", "million", "billion".
    4. If 'scale' is "", 'ans' must be a fully-resolved number (no remaining scaling).
    
    Output Format:
    - The LAST line of the program MUST be: print(json.dumps({"ans": ans, "scale": scale}, ensure_ascii=False))
    - Output raw Python only. No Markdown. No other print statements.
    """
    table = dspy.InputField(desc="Financial table in markdown/text format")
    paragraphs = dspy.InputField(desc="Supporting text context")
    question = dspy.InputField(desc="The question to answer")

    program = dspy.OutputField(desc="Python code ending with JSON print of ans and scale")

# evaluate the code
def normalize_scale(s):
    if s is None: return ""
    # Convert to string and aggressively strip whitespace and all literal quotes
    s = str(s).strip().replace('"', '').replace("'", "")
    
    s = s.lower()
    allowed = {"", "percent", "thousand", "million", "billion"}
    return s if s in allowed else ""

# load the data

tatqa_metric = TaTQAEmAndF1()

def validate_code_metric(gold, pred, trace=None):
    # Execute the code
    stdout, error, status = exec_with_timeout(pred.program)
    lines = [ln for ln in stdout.strip().split('\n') if ln.strip()]
    stdout_last = lines[-1] if lines else ""

    # Extract answer and scale from the JSON output 
    try:
        parsed_output = json.loads(stdout_last)
        # format: {"ans": ..., "scale": ...}
        final_pred_ans = parsed_output.get("ans", stdout_last)
        raw_scale = parsed_output.get("scale", "")
    except json.JSONDecodeError:
        final_pred_ans = stdout_last
        raw_scale = ""

    # Normalize for the evaluator
    pred_scale_norm = normalize_scale(raw_scale)
    gold_scale_norm = normalize_scale(getattr(gold, "scale", ""))

    # Ground Truth for the official metric (use real answer_type from dataset)
    ground_truth = {
        "answer_type": getattr(gold, "answer_type", "arithmetic"),
        "scale": gold_scale_norm,
        "answer": getattr(gold, "answer", None),
    }

    # Run official TaTQA Evaluation
    tatqa_metric(ground_truth, final_pred_ans, pred_scale=pred_scale_norm)
    em, f1, scale_score, op_score = tatqa_metric.get_overall_metric(reset=True)

    score = 0.5 * em + 0.5 * f1
    passed_strict = (em == 1.0 and scale_score == 1.0)

    log_jsonl({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "qid": getattr(gold, "qid", None),
        "gold_scale": gold_scale_norm,
        "pred_scale": pred_scale_norm,
        "gold_answer": getattr(gold, "answer", None),
        "pred_answer": final_pred_ans,
        "stdout_last": stdout_last,
        "status": status,
        "exec_error": error,
        "passed_strict": passed_strict,
        "score": score,
        "em": em,
        "f1": f1,
        "scale_score": scale_score,
    })
    return (score, passed_strict)



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
        answer_type_val = q_obj.get("answer_type", "arithmetic")
        derivation_val = q_obj.get("derivation", "")

        example = dspy.Example(
            qid=q_obj.get("uid"),
            table=table_text,
            paragraphs=paragraphs_text,
            question=q_obj.get("question", ""),
            scale=scale_val,
            answer=q_obj.get("answer"),
            answer_type=answer_type_val,
            derivation=derivation_val,
        ).with_inputs("table", "paragraphs", "question")

        dataset.append(example)

    return dataset

if __name__ == '__main__':
    trainset = load_data(SAMPLE_PATH)

    # initialize MIPROv2 (let 'auto' choose search size)
    # MIPRO maximizes the returned value; use continuous score 0.5*em + 0.5*f1
    teleprompter = MIPROv2(metric=lambda g, p, t=None: validate_code_metric(g, p, t)[0], auto="light")

    # run the "Prompt Battle"
    optimized_program = teleprompter.compile(
        dspy.Predict(FinancialCodegen), 
        trainset=trainset,
        max_bootstrapped_demos=3, # Pick 3 best examples to put in prompt
        max_labeled_demos=2       # Pick 2 gold examples
    )

    out_path = Path(__file__).resolve().parent / "best_financial_prompt.json"
    optimized_program.save(out_path)

    if hasattr(teleprompter, 'trials_df'):
        print("\n--- Optimization Results ---")
        # Displaying the top trials by score
        df = teleprompter.trials_df.sort_values(by='score', ascending=False)
        print(df[['instruction', 'score']].head(10))
        