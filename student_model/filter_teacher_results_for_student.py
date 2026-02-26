import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "evaluation_metrics"))
from tatqa_metric import TaTQAEmAndF1

DATA_DIR = PROJECT_ROOT / "dataset_filtered"
TRAIN_PATH = DATA_DIR / "tatqa_dataset_train_filtered.json"


def normalize_scale(s):
    if s is None:
        return ""
    s = str(s).strip().replace('"', "").replace("'", "").lower()
    return s if s in {"", "percent", "thousand", "million", "billion"} else ""


def build_qid_to_context_and_gold(train_path: Path):
    data = json.loads(train_path.read_text(encoding="utf-8"))
    qid_to_ctx = {}

    for entry in data:
        table_rows = entry.get("table", {}).get("table", [])
        table_text = "\n".join(" | ".join(map(str, row)) for row in table_rows)
        paras_text = "\n".join(p.get("text", "") for p in entry.get("paragraphs", []))

        for q in entry.get("questions", []):
            qid = str(q.get("uid", ""))
            if not qid: continue

            qid_to_ctx[qid] = {
                "table": table_text,
                "paragraphs": paras_text,
                "question": q.get("question", ""),
                "answer_type": q.get("answer_type", "arithmetic"),
                "scale": normalize_scale(q.get("scale", "")),
                "answer": q.get("answer"),
            }
            
    return qid_to_ctx


def pred_ans_scale_from_row(row):
    if row.get("pred_ans") is not None:
        return row.get("pred_ans"), normalize_scale(row.get("pred_scale", ""))
    
    last = row.get("stdout_last")
    if not last: return None, ""
    
    try:
        obj = json.loads(last)
        return obj.get("ans"), normalize_scale(obj.get("scale", ""))
    except:
        return last, ""


def main():
    ap = argparse.ArgumentParser(description="Filter teacher results for student fine-tuning")
    ap.add_argument("--in_jsonl", type=str, default="student_model/teacher_full_train/teacher_codegen_train_run.jsonl", help="Teacher train RUN results JSONL (has run_status, pred_ans, stdout_last)")
    ap.add_argument("--out_jsonl", type=str, default="student_model/teacher_full_train/student_train_from_teacher.jsonl", help="Output JSONL for student training")
    ap.add_argument("--train_json", type=str, default=None, help="Train filtered JSON (default: dataset_filtered/tatqa_dataset_train_filtered.json)")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_json) if args.train_json else TRAIN_PATH
    if not train_path.exists():
        raise FileNotFoundError(f"Train JSON not found: {train_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Teacher results not found: {in_path}")


    qid_to_ctx = build_qid_to_context_and_gold(train_path)
    metric = TaTQAEmAndF1()

    retained = []
    n_ok = n_correct = 0
    for line in in_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("qid")
        if not qid or qid not in qid_to_ctx:
            continue

        # Must have execution success 
        if row.get("run_status") != "ok" or not (row.get("generated_code") or "").strip():
            continue
        n_ok += 1

        # and have answer correctness — TaTQA EM 
        ctx = qid_to_ctx[qid]
        ground_truth = {"answer_type": ctx["answer_type"], "scale": ctx["scale"], "answer": ctx["answer"]}
        pred_ans, pred_scale = pred_ans_scale_from_row(row)
        metric(ground_truth, pred_ans if pred_ans is not None else [], pred_scale=pred_scale)
        em, _, _, _ = metric.get_overall_metric(reset=True)
        if em < 1.0:
            continue
        n_correct += 1

        out_row = {
            "qid": qid,
            "table": ctx["table"],
            "paragraphs": ctx["paragraphs"],
            "question": ctx["question"],
            "generated_code": row["generated_code"],
            "gold_answer": ctx["answer"],
            "gold_scale": ctx["scale"],
        }
        retained.append(out_row)

    with out_path.open("w", encoding="utf-8") as f:
        for r in retained:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Teacher results: execution ok = {n_ok}, of those answer correct (EM) = {n_correct}, retained = {len(retained)} (both conditions)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
