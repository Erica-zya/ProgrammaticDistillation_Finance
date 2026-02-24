import os
import json
from pathlib import Path

from dotenv import load_dotenv
import dspy
from dspy.teleprompt import MIPROv2

from teacher_model.run_generated_code import exec_with_timeout, to_float_maybe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = Path(__file__).resolve().parent / "sample_of_each_type.json"
DATA_DIR = PROJECT_ROOT / "dataset_filtered"

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
    Rules:
    - Normalization: Final print() MUST be a raw number only.
    - Percentages: Convert to decimals (e.g., print 0.05 for 5%).
    - Scale: Handle 'millions'/'billions' in-code (e.g., value * 1e6).
    - Extraction: Use clean_val() for all table/text parsing.
    """
    table = dspy.InputField(desc="Financial table in markdown/text format")
    paragraphs = dspy.InputField(desc="Supporting text context")
    question = dspy.InputField(desc="The question to answer")
    scale = dspy.InputField(desc="Scale label for the answer (e.g., ones, thousand, million, billions, percent)")
    program = dspy.OutputField(desc="Python code that prints the normalized numeric answer")

# evaluate the code (placeholder for now)
def validate_code_metric(gold, pred, trace=None):
    stdout, error, status = exec_with_timeout(pred.program)
    
    # Get the last line of output
    lines = [ln for ln in stdout.strip().split('\n') if ln.strip()]
    if not lines: return False
    
    pred_val = to_float_maybe(lines[-1])
    gold_val = to_float_maybe(gold.answer)
    
    if pred_val is not None and gold_val is not None:
        # Check if gold is a percentage (often > 1 while normalized pred is < 1)
        # Or if the gold answer string contains '%'
        is_gold_pct = (gold_val > 1 and pred_val < 1 and abs(pred_val * 100 - gold_val) < 1e-2)
        
        if is_gold_pct or abs(pred_val - gold_val) < 1e-4:
            return True
    return False

# load the data
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
        scale_val = q_obj.get("scale", "ones")

        example = dspy.Example(
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
        