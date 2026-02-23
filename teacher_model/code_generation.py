import os, json, time, argparse, re
from pathlib import Path
import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset_filtered"
OUT_DIR = SCRIPT_DIR / "outputs"

SPLIT_MAP = {
    "train": "tatqa_dataset_train_filtered.json",
    "dev": "tatqa_dataset_dev_filtered.json",
    "test": "tatqa_dataset_test_gold_filtered.json"
}

PROMPT_TMPL = """Write a Python program that answers the QUESTION using only TABLE and PARAGRAPHS.
Rules:
- Use ONLY the given TABLE and PARAGRAPHS.
- Infer and apply any scale/unit ONLY if explicitly stated.
- Output MUST be raw Python code only.
- End with a single print(...) of the final answer.

TABLE:\n{table}\nPARAGRAPHS:\n{paras}\nQUESTION:\n{question}\nGOLD_DERIVATION:\n{derivation}"""



def call_hf_chat(url, token, model, prompt, max_retries=8):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"model": model, "messages": [
        {"role": "system", "content": "Return ONLY raw Python code. No Markdown fences."},
        {"role": "user", "content": prompt}
    ], "temperature": 0.0, "max_tokens": 512}
    
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return re.sub(r"^```(?:python)?\s*\n([\s\S]*?)\n```$", r"\1", content.strip(), flags=re.IGNORECASE).strip()
        except requests.RequestException:
            time.sleep(1.5 * (2 ** attempt))
    return ""




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=list(SPLIT_MAP.keys()))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    load_dotenv()

    token, model = os.getenv("HF_TOKEN"), os.getenv("HF_MODEL_72B", "Qwen/Qwen2.5-72B-Instruct")
    url = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")
    
    in_path = DATA_DIR / SPLIT_MAP[args.split]
    out_path = OUT_DIR / f"teacher_codegen_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = {str(json.loads(line)["qid"]) for line in out_path.open() if line.strip()} if out_path.exists() else set()
    data = json.loads(in_path.read_text(encoding="utf-8"))

    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for doc in data:
            doc_id = doc.get("table", {}).get("uid", "unknown")
            table_text = "\n".join(" | ".join(map(str, r)) for r in doc.get("table", {}).get("table", []))
            paras_text = "\n".join(p.get("text", "") for p in doc.get("paragraphs", []))

            for qi, q in enumerate(doc.get("questions", [])):
                if args.limit and written >= args.limit:
                    print(f"[DONE] Reached limit of {args.limit} samples.")
                    return

                qid = str(q.get("uid", f"{doc_id}_{qi}"))
                if qid in done: continue

                prompt = PROMPT_TMPL.format(table=table_text, paras=paras_text, question=q.get("question", ""), derivation=q.get("derivation", ""))
                t0 = time.time()
                try:
                    code = call_hf_chat(url, token, model, prompt)
                    rec = {"qid": qid, "split": args.split, "doc_id": doc_id, "gold_answer": q.get("answer"), "teacher_model": model, "latency_sec": round(time.time() - t0, 3), "generated_code": code, "status": "ok"}
                except Exception as e:
                    rec = {"qid": qid, "status": "api_error", "error": str(e)}

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if written % 10 == 0: print(f"Processed {written} samples...")

if __name__ == "__main__":
    main()