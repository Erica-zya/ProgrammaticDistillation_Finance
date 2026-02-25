#!/usr/bin/python
import argparse  # Python standard library: used to parse command-line arguments (e.g., --gold_path).
import json # Standard library: used to read/write JSON files.
from tatqa_metric import * # Import all public names from tatqa_metric (excluding private ones).
from typing import Any, Dict, Tuple # Only for type annotations (does not affect runtime).

#######################
# code from TAT-QA repo
########################


# predicted_answers is a dictionary:
#   question_id -> [answer, scale]
# golden_answers is essentially a list of contexts (each containing questions). (the original dataset)
#
# em_and_f1(
#     ground_truth = qa,          # A single question dictionary from the gold file
#     prediction   = pred_answer, # The predicted answer for this question
#     pred_scale   = pred_scale   # The predicted scale for this question
# )

def evaluate_json(golden_answers: Dict[str, Any], predicted_answers: Dict[str, Any]) -> Tuple[float, float]:
    # evaluate_json() drives the evaluation process:
    # iterate over every question in the gold data → 
    # retrieve the corresponding prediction → 
    # feed them into TaTQAEmAndF1 to accumulate statistics → 
    # finally compute overall metrics.
    em_and_f1 = TaTQAEmAndF1()
    em_and_f1_sub = TaTQAEmAndF1()
    # This is not a pure function that returns a score per question.
    # Instead, it acts as an accumulator:
    # feed questions one by one → internally update statistics →
    # retrieve overall / detailed metrics at the end.
    hit = 0
    total = 0
    for qas in golden_answers:
        for qa in qas["questions"]:
            total += 1
            query_id = qa["uid"]
            pred_answer, pred_scale = None, None
            if query_id in predicted_answers:
                hit += 1
                pred_answer, pred_scale = predicted_answers[query_id]
                em_and_f1_sub(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)
            em_and_f1(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)

    print("pred coverage:", hit, "/", total)

    # Retrieve overall metrics after processing all questions
    global_em, global_f1, global_scale, global_op = em_and_f1.get_overall_metric()
    # ---- SUBSET metrics (only predicted questions) ----
    sub_em, sub_f1, sub_scale, sub_op = em_and_f1_sub.get_overall_metric()
    print(f"[Coverage] Predicted questions: {hit}/{total} ({hit/total*100:.2f}%)")

    # Overall metrics (all questions; missing preds treated as wrong)
    # # Subset metrics (only questions that have predictions)
    sub_em, sub_f1, sub_scale, sub_op = em_and_f1_sub.get_overall_metric()

    print("\n==================== Results: SUBSET (predicted only) ====================")
    print("EM     : {0:.2f}".format(sub_em * 100))
    print("F1     : {0:.2f}".format(sub_f1 * 100))
    print("Scale  : {0:.2f}".format(sub_scale * 100))
    print("Table  : {0:.2f}   &   {1:.2f}".format(sub_em * 100, sub_f1 * 100))

    print("\n[SUBSET] Breakdown (count by answer_type × answer_from)")
    detail_raw_sub = em_and_f1_sub.get_raw_pivot_table()
    print(detail_raw_sub)

    detail_em_sub, detail_f1_sub = em_and_f1_sub.get_detail_metric()
    print("\n[SUBSET] EM breakdown (mean by answer_type × answer_from)")
    print(detail_em_sub)
    print("\n[SUBSET] F1 breakdown (mean by answer_type × answer_from)")
    print(detail_f1_sub)

    print("\n==================== Results: FULL SET (missing preds = wrong) ====================")
    print("EM     : {0:.2f}".format(global_em * 100))
    print("F1     : {0:.2f}".format(global_f1 * 100))
    print("Scale  : {0:.2f}".format(global_scale * 100))
    print("Table  : {0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))

    print("\n[FULL] Breakdown (count by answer_type × answer_from)")
    detail_raw = em_and_f1.get_raw_pivot_table()
    print(detail_raw)

    detail_em, detail_f1 = em_and_f1.get_detail_metric()
    print("\n[FULL] EM breakdown (mean by answer_type × answer_from)")
    print(detail_em)
    print("\n[FULL] F1 breakdown (mean by answer_type × answer_from)")
    print(detail_f1)

def save_json(obj, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# Load gold and prediction files, then call evaluate_json
def evaluate_prediction_file(gold_path: str,
                             pred_path: str):
    


    """
    Evaluate a prediction file against the gold file.

    Notes:
    - Add an intermediate "evaluation input" file generated from the JSONL predictions.
      This file is saved next to pred_path with suffix "_eval_input.json".
    """


    golden_answers = json.load(open(gold_path, encoding='utf-8'))

    # Load gold answers (do not change)
    with open(pred_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    
    
    # Load predictions from JSONL (one JSON object per line)
    pred_map={}
    for r in results:
        pred_map[r["qid"]]=[r["pred_answer"],""]

    # Construct the dict expected by the evaluator:
    # { qid: [pred_answer, ""] }
    save_path=pred_path[:-6]+"_eval_input.json"
    # Save an intermediate file for debugging / reproducibility
    # Example: teacher_codegen_test_results.jsonl -> teacher_codegen_test_results_eval_input.json

    save_json(pred_map, save_path)
    
    # Run evaluation
    evaluate_json(golden_answers, pred_map)


if __name__ == "__main__":
    # pylint: disable=invalid-name
    # How to use:
    # python3 ./test_metric/tatqa_eval.py --gold_path ./dataset_raw/tatqa_dataset_test_gold.json --pred_path ./teacher_model/outputs/teacher_codegen_test_results.jsonl
    parser = argparse.ArgumentParser(description='evaluation on TAT-QA dataset')
    parser.add_argument("--gold_path",
                        type=str,
                        required=True,
                        default="tatqa_dataset_test_gold.json",
                        help='The path of the gold file')
    parser.add_argument("--pred_path",
                        type=str,
                        required=True,
                        default="sample_predictions.json",
                        help='The path of the prediction file')
    
    args = parser.parse_args()
    evaluate_prediction_file(args.gold_path, args.pred_path)
