#!/usr/bin/python
"""
TAT-QA evaluation for subset runs (e.g. first 500 train).
Filters gold to only questions that have predictions, then runs the same
TaTQA EM/F1/scale metrics. Use when pred_path has fewer samples than the full gold file.
"""
import argparse
import json
from tatqa_metric import TaTQAEmAndF1
from typing import Any, Dict

# predicted_answers: question_id -> [answer, scale]
# golden_answers: list of contexts, each with "questions" (each qa has "uid", etc.)


def evaluate_json(golden_answers: list, predicted_answers: Dict[str, Any]) -> None:
    em_and_f1 = TaTQAEmAndF1()
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
                em_and_f1(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)

    print("pred coverage:", hit, "/", total)
    print(f"[Coverage] Predicted questions: {hit}/{total} ({hit/total*100:.2f}%)")

    em, f1, scale_metric, _ = em_and_f1.get_overall_metric()
    print("\n==================== Results (subset: predicted only) ====================")
    print("EM     : {0:.2f}".format(em * 100))
    print("F1     : {0:.2f}".format(f1 * 100))
    print("Scale  : {0:.2f}".format(scale_metric * 100))
    print("Table  : {0:.2f}   &   {1:.2f}".format(em * 100, f1 * 100))

    print("\n[Breakdown] count by answer_type × answer_from")
    print(em_and_f1.get_raw_pivot_table())
    detail_em, detail_f1 = em_and_f1.get_detail_metric()
    print("\n[EM breakdown] mean by answer_type × answer_from")
    print(detail_em)
    print("\n[F1 breakdown] mean by answer_type × answer_from")
    print(detail_f1)


def save_json(obj: Any, save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def evaluate_prediction_file(gold_path: str, pred_path: str) -> None:
    """
    Load gold and predictions; filter gold to only questions that have predictions;
    run TAT-QA evaluation on that subset (e.g. for first 500 train results).
    """
    golden_answers = json.load(open(gold_path, encoding="utf-8"))
    with open(pred_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    pred_map = {}
    for r in results:
        pred_map[r["qid"]] = [r["pred_ans"], r["pred_scale"]]

    save_path = (pred_path[:-6] if pred_path.endswith(".jsonl") else pred_path) + "_eval_input.json"
    save_json(pred_map, save_path)

    pred_qids = set(pred_map.keys())
    golden_subset = []
    for ctx in golden_answers:
        qs = [qa for qa in ctx["questions"] if qa["uid"] in pred_qids]
        if qs:
            golden_subset.append({**ctx, "questions": qs})

    print(f"[Subset] Evaluating {len(pred_map)} predictions (gold filtered to matching questions).\n")
    evaluate_json(golden_subset, pred_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TAT-QA evaluation on a subset (e.g. first 500 train). Gold is filtered to predicted qids."
    )
    parser.add_argument("--gold_path", type=str, required=True, help="Path to gold JSON (e.g. dataset_filtered/tatqa_dataset_train_filtered.json)")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to prediction JSONL (qid, pred_ans, pred_scale per line)")
    args = parser.parse_args()
    evaluate_prediction_file(args.gold_path, args.pred_path)
