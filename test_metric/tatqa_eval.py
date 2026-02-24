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
    # This is not a pure function that returns a score per question.
    # Instead, it acts as an accumulator:
    # feed questions one by one → internally update statistics →
    # retrieve overall / detailed metrics at the end.
    for qas in golden_answers:
        for qa in qas["questions"]:
            query_id = qa["uid"]
            pred_answer, pred_scale = None, None
            if query_id in predicted_answers:
                pred_answer, pred_scale = predicted_answers[query_id]
            em_and_f1(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)

    # Retrieve overall metrics after processing all questions
    global_em, global_f1, global_scale, global_op = em_and_f1.get_overall_metric()
    print("----")
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("Scale score {0:.2f}".format(global_scale * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")

    detail_raw = em_and_f1.get_raw_pivot_table()
    print("---- raw detail ---")
    print(detail_raw)
    detail_em, detail_f1 = em_and_f1.get_detail_metric()
    print("---- em detail ---")
    print(detail_em)
    print("---- f1 detail ---")
    print(detail_f1)



# Load gold and prediction files, then call evaluate_json
def evaluate_prediction_file(gold_path: str,
                             pred_path: str):

    golden_answers = json.load(open(gold_path, encoding='utf-8'))
    predicted_answers = json.load(open(pred_path, encoding='utf-8'))
    evaluate_json(golden_answers, predicted_answers)


if __name__ == "__main__":
    # pylint: disable=invalid-name
    # How to use:
    # python3 tatqa_eval.py --gold_path ../dataset_raw/tatqa_dataset_dev.json --pred_path ./sample_prediction.json
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
