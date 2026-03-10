import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset_filtered"
OUT_DIR = SCRIPT_DIR

SPLIT_MAP = {
    "train": "tatqa_dataset_train_filtered.json",
    "dev": "tatqa_dataset_dev_filtered.json",
    "test": "tatqa_dataset_test_filtered.json",
}


def safe_str(x):
    return "" if x is None else str(x)


def flatten_split(split="dev"):
    in_path = DATA_DIR / SPLIT_MAP[split]
    out_path = OUT_DIR / f"{split}_flattened.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for doc in data:
            doc_id = doc.get("table", {}).get("uid", "unknown")

            # 跟你原来 pipeline 保持一致的 table 处理
            table_text = "\n".join(" | ".join(map(str, r)) for r in doc.get("table", {}).get("table", []))

            paras_text = "\n".join(p.get("text", "") for p in doc.get("paragraphs", []))

            for qi, q in enumerate(doc.get("questions", [])):
                #qid = str(q.get("uid", f"{doc_id}_{qi}"))
                qid=str(q["uid"])
                rec = {
                    "qid": qid,
                    "table": table_text,
                    "paragraphs": paras_text,
                    "question": q.get("question", ""),
                    "gold_answer": q.get("answer"),
                    "gold_scale": q.get("scale", ""),
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Done. Wrote {written} examples to {out_path}")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def compare_train(
    student_path,
    flattened_path,
    max_show=10,
):
    student_path = Path(student_path)
    flattened_path = Path(flattened_path)

    student_data = load_jsonl(student_path)
    flattened_data = load_jsonl(flattened_path)

    flat_by_qid = {str(x["qid"]): x for x in flattened_data}

    total = 0
    missing_in_flattened = []
    table_mismatches = []
    paragraph_mismatches = []

    for s in student_data:
        qid = str(s["qid"])
        total += 1

        if qid not in flat_by_qid:
            missing_in_flattened.append(qid)
            continue

        f = flat_by_qid[qid]

        student_table = s.get("table", "")
        flat_table = f.get("table", "")

        student_paragraphs = s.get("paragraphs", "")
        flat_paragraphs = f.get("paragraphs", "")

        if student_table != flat_table:
            table_mismatches.append(qid)

        if student_paragraphs != flat_paragraphs:
            paragraph_mismatches.append(qid)

    print(f"Total student records checked: {total}")
    print(f"Missing in flattened: {len(missing_in_flattened)}")
    print(f"Table mismatches: {len(table_mismatches)}")
    print(f"Paragraph mismatches: {len(paragraph_mismatches)}")

    if missing_in_flattened:
        print("\nMissing qids (first few):")
        for qid in missing_in_flattened[:max_show]:
            print(qid)

    if table_mismatches:
        print("\nTable mismatch qids (first few):")
        for qid in table_mismatches[:max_show]:
            print(qid)

    if paragraph_mismatches:
        print("\nParagraph mismatch qids (first few):")
        for qid in paragraph_mismatches[:max_show]:
            print(qid)

    if (
        len(missing_in_flattened) == 0
        and len(table_mismatches) == 0
        and len(paragraph_mismatches) == 0
    ):
        print("\nAll matched exactly.")

if __name__ == "__main__":
    #flatten_split("dev")
    #flatten_split("test")
    #flatten_split("train")

    train_correct_path=SCRIPT_DIR.parent / "student_model_fine_tune/teacher_full_train/student_train_from_teacher.jsonl"
    train_wrong_path=SCRIPT_DIR.parent / "student_model_fine_tune/teacher_full_train/student_train_from_teacher_round2_wrong.jsonl"
    flat_path="./train_flattened.jsonl"
    compare_train(student_path=train_correct_path,flattened_path=flat_path)
    compare_train(student_path=train_wrong_path,flattened_path=flat_path)