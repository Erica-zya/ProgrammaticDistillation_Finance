import json
from pathlib import Path

# ============================================================
# Filter utilities for the raw TAT-QA dataset (context-level JSON)
#
# Each JSON file is a list of "contexts". Each context contains:
#   - table: { "uid": str, "table": List[List[str]] }
#   - paragraphs: List[ { "uid": str, "order": int, "text": str } ]
#   - questions: List[ question objects ... ]
#
# Goal:
#   1) Load the raw JSON
#   2) Remove question samples whose "derivation" field is empty
#   3) Save a new filtered JSON file
#   4) Print summary stats (before vs after)
# ============================================================


def count_total_questions(data):
    # Count total number of question samples across all contexts
    total = 0
    for ctx in data:
        total += len(ctx.get("questions", []))
    return total





def filter_tatqa_remove_empty_derivation(input_path: str,output_path: str):
    """
    Read a TAT-QA JSON file,
    remove question samples with empty derivation, and save a new JSON.

    Args:
        input_path: path to raw JSON
        output_path: path to save filtered JSON
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Load data: a list of contexts
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of contexts.")
    
    filtered_data = []
    # Iterate contexts

    n_ctx_before=len(data)
    n_q_before = count_total_questions(data)
    removed_q = 0

    for ctx in data:
        # Copy context dict (so we don't mutate original object reference)
        new_ctx = dict(ctx)

        questions = ctx["questions"]

        new_questions=[]

        for q in questions:
            #print("=====")
            d = q.get("derivation", "")
            if isinstance(d, str) and d.strip() != "":
                new_questions.append(q)
            else:
                removed_q += 1
        if new_questions==[]:
            continue
        new_ctx["questions"]=new_questions
        filtered_data.append(new_ctx)
        
    #print(filtered_data)
    
    n_ctx_after=len(filtered_data)
    n_q_after = count_total_questions(filtered_data)
    Path(output_path).write_text(json.dumps(filtered_data, ensure_ascii=False, indent=2), encoding="utf-8")


    # Print summary
    print("\n" + "=" * 80)
    print("Filter summary: remove samples with empty derivation")
    print("-" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Contexts:  {n_ctx_before} -> {n_ctx_after}")
    print(f"Questions: {n_q_before} -> {n_q_after}")

    keep_rate = (n_q_after / n_q_before) if n_q_before > 0 else 0.0
    print(f"Removed questions: {removed_q} ({(removed_q / n_q_before * 100.0) if n_q_before > 0 else 0.0:.2f}%)")
    print(f"Keep rate: {keep_rate * 100.0:.2f}%")
    print("=" * 80 + "\n")

    return filtered_data



    




if __name__ == "__main__":
    # Input & Output path
    input_json = "../dataset_raw/tatqa_dataset_test_gold.json"
    output_json = "../dataset_filtered/tatqa_dataset_test_gold_filtered.json"
    filter_tatqa_remove_empty_derivation(input_path=input_json,output_path=output_json)
    input_json = "../dataset_raw/tatqa_dataset_train.json"
    output_json = "../dataset_filtered/tatqa_dataset_train_filtered.json"
    filter_tatqa_remove_empty_derivation(input_path=input_json,output_path=output_json)
    input_json = "../dataset_raw/tatqa_dataset_dev.json"
    output_json = "../dataset_filtered/tatqa_dataset_dev_filtered.json"
    filter_tatqa_remove_empty_derivation(input_path=input_json,output_path=output_json)


