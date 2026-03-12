import json
import argparse
from pathlib import Path


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_train", type=str, required=True,
                    help="Original student_train_from_teacher jsonl")
    ap.add_argument("--add_train", type=str, required=True,
                    help="New EM=1 jsonl from train_wrong")
    ap.add_argument("--out_path", type=str, required=True,
                    help="Merged output path")
    args = ap.parse_args()

    base_rows = load_jsonl(args.base_train)
    add_rows = load_jsonl(args.add_train)

    merged = []
    seen = set()

    # keep original base rows first
    for r in base_rows:
        qid = str(r["qid"])
        if qid not in seen:
            merged.append(r)
            seen.add(qid)

    added_count = 0
    skipped_count = 0

    # append new EM=1 rows if qid not already present
    for r in add_rows:
        qid = str(r["qid"])
        if qid not in seen:
            merged.append(r)
            seen.add(qid)
            added_count += 1
        else:
            skipped_count += 1

    save_jsonl(merged, args.out_path)

    print(f"Base rows: {len(base_rows)}")
    print(f"Add rows: {len(add_rows)}")
    print(f"Actually added: {added_count}")
    print(f"Skipped due to duplicate qid: {skipped_count}")
    print(f"Final merged rows: {len(merged)}")
    print(f"Saved to: {args.out_path}")


if __name__ == "__main__":
    main()