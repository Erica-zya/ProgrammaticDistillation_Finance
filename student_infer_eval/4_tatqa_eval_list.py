#!/usr/bin/python
import argparse
import json
import re
from tatqa_metric import *
from typing import Any, Dict, Tuple


def evaluate_json(golden_answers: Dict[str, Any], predicted_answers: Dict[str, Any]) -> Tuple[float, float]:
    em_and_f1 = TaTQAEmAndF1()
    em_and_f1_sub = TaTQAEmAndF1()

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

    global_em, global_f1, global_scale, global_op = em_and_f1.get_overall_metric()
    sub_em, sub_f1, sub_scale, sub_op = em_and_f1_sub.get_overall_metric()
    print(f"[Coverage] Predicted questions: {hit}/{total} ({hit/total*100:.2f}%)")

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


def extract_candidate_indices(record):
    """
    Supports keys like:
      pred_answer_1, pred_scale_1
      pred_answer_2, pred_scale_2
      ...
    """
    indices = set()
    for k in record.keys():
        m = re.fullmatch(r"pred_answer_(\d+)", k)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def score_one_candidate(qa, pred_answer, pred_scale):
    """
    Run the official metric on one candidate only, and return overall metrics
    for this single example.
    """
    meter = TaTQAEmAndF1()
    meter(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)
    em, f1, scale, op = meter.get_overall_metric()
    return em, f1, scale, op


def evaluate_prediction_file(gold_path: str, pred_path: str):
    golden_answers = json.load(open(gold_path, encoding="utf-8"))

    with open(pred_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    # Build gold lookup: qid -> gold question dict
    gold_by_qid = {}
    for qas in golden_answers:
        for qa in qas["questions"]:
            gold_by_qid[str(qa["uid"])] = qa

    pred_map = {}
    best_meta = {}
    oracle_hit = 0
    oracle_total = 0
    #best_tuple = None
    #best_pair = [None, None]
    #best_idx = None

    for r in results:
        qid = str(r["qid"])
        if qid not in gold_by_qid:
            continue

        qa = gold_by_qid[qid]
        oracle_total += 1

        candidate_indices = extract_candidate_indices(r)

        # backward compatibility:
        # if file only has pred_answer / pred_scale (single prediction)
        if not candidate_indices and "pred_answer" in r:
            pred_map[qid] = [r.get("pred_answer"), r.get("pred_scale")]
            best_meta[qid] = {
                "best_idx": None,
                "best_pred_answer": r.get("pred_answer"),
                "best_pred_scale": r.get("pred_scale"),
                }
            continue

        best_tuple = None
        best_pair = [None, None]
        best_idx = None

        for i in candidate_indices:
            pred_answer = r.get(f"pred_answer_{i}")
            pred_scale = r.get(f"pred_scale_{i}", "")

            em, f1, scale, op = score_one_candidate(qa, pred_answer, pred_scale)
            score_tuple = (em, f1, scale)   # EM first, then F1, then scale

            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best_pair = [pred_answer, pred_scale]
                best_idx = i

        pred_map[qid] = best_pair

        best_meta[qid] = {
            "best_idx": best_idx,
            "best_pred_answer": best_pair[0],
            "best_pred_scale": best_pair[1],
            }

        if best_tuple is not None and (best_tuple[0] > 0 or best_tuple[1] > 0):
            oracle_hit += 1

    print(f"Oracle hit (>0 EM or F1): {oracle_hit}/{oracle_total} ({oracle_hit/oracle_total*100:.2f}%)")

    save_path = pred_path[:-6] + "_eval_input.json"
    save_json(pred_map, save_path)
    save_json(best_meta, pred_path[:-6] + "_best_candidate.json")

    evaluate_json(golden_answers, pred_map)


if __name__ == "__main__":
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