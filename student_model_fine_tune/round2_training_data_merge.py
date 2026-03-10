import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

orig_path = SCRIPT_DIR / "training_data" / "student_train_from_teacher.jsonl"
recovered_path = SCRIPT_DIR / "recover_round1" / "recovered_successes.jsonl"

out_dir = SCRIPT_DIR / "training_data"
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "student_train_from_teacher_round2.jsonl"

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

orig_data = load_jsonl(orig_path)
recovered_data = load_jsonl(recovered_path)

seen_qids = {x["qid"] for x in orig_data if "qid" in x}

new_recovered = [x for x in recovered_data if x.get("qid") not in seen_qids]

merged = orig_data + new_recovered

with out_path.open("w", encoding="utf-8") as f:
    for row in merged:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Original samples: {len(orig_data)}")
print(f"Recovered samples: {len(recovered_data)}")
print(f"New recovered added: {len(new_recovered)}")
print(f"Round2 total samples: {len(merged)}")
print(f"Saved to: {out_path}")