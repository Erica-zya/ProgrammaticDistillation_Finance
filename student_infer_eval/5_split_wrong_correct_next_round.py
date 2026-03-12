#!/usr/bin/python
import argparse
import json
import re
from pathlib import Path
from tatqa_metric import TaTQAEmAndF1


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_gold_lookup(gold_path):
    gold_data = json.load(open(gold_path, encoding="utf-8"))
    gold_by_qid = {}
    for doc in gold_data:
        for qa in doc["questions"]:
            gold_by_qid[str(qa["uid"])] = qa
    return gold_by_qid


def extract_candidate_indices(record):
    indices = set()
    for k in record.keys():
        m = re.fullmatch(r"pred_answer_(\d+)", k)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def score_one_candidate(qa, pred_answer, pred_scale):
    meter = TaTQAEmAndF1()
    meter(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)
    em, f1, scale, op = meter.get_overall_metric()
    return em, f1, scale, op


def build_output_record(row, generated_code):
    return {
        "qid": row["qid"],
        "table": row["table"],
        "paragraphs": row["paragraphs"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "gold_scale": row.get("gold_scale", ""),
        "generated_code": generated_code,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_path", type=str, required=True)
    ap.add_argument("--pred_path", type=str, required=True)
    ap.add_argument("--out_em1", type=str, default=None)
    ap.add_argument("--out_rest", type=str, default=None)
    args = ap.parse_args()

    pred_path = Path(args.pred_path)
    out_em1 = args.out_em1 or str(pred_path.with_name(pred_path.stem + "_em1.jsonl"))
    out_rest = args.out_rest or str(pred_path.with_name(pred_path.stem + "_rest.jsonl"))

    gold_by_qid = build_gold_lookup(args.gold_path)
    rows = load_jsonl(args.pred_path)

    em1_rows = []
    rest_rows = []

    total = 0
    em1_count = 0
    rest_count = 0

    for row in rows:
        qid = str(row["qid"])
        if qid not in gold_by_qid:
            continue

        total += 1
        qa = gold_by_qid[qid]
        candidate_indices = extract_candidate_indices(row)

        if not candidate_indices:
            continue

        chosen_em1_code = None

        for i in candidate_indices:
            pred_answer = row.get(f"pred_answer_{i}")
            pred_scale = row.get(f"pred_scale_{i}", "")
            generated_code = row.get(f"generated_code_{i}", "")

            em, f1, scale, op = score_one_candidate(qa, pred_answer, pred_scale)

            if em == 1.0:
                chosen_em1_code = generated_code
                break

        if chosen_em1_code is not None:
            em1_count += 1
            em1_rows.append(build_output_record(row, chosen_em1_code))
        else:
            last_idx = candidate_indices[-1]
            last_code = row.get(f"generated_code_{last_idx}", "")
            rest_count += 1
            rest_rows.append(build_output_record(row, last_code))

    save_jsonl(em1_rows, out_em1)
    save_jsonl(rest_rows, out_rest)

    print(f"Total rows checked: {total}")
    print(f"Rows with at least one EM=1 candidate: {em1_count}")
    print(f"Rows with no EM=1 candidate: {rest_count}")
    print(f"Saved EM=1 rows to: {out_em1}")
    print(f"Saved rest rows to: {out_rest}")


if __name__ == "__main__":
    main()