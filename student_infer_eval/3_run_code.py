import json
import re
import math
import argparse
import io
import contextlib
import multiprocessing as mp
from pathlib import Path


def to_float_maybe(x):
    try:
        s = str(x).strip().replace(",", "").replace("$", "").replace("%", "")
        if re.fullmatch(r"\(\s*-?\d+(\.\d+)?\s*\)", s):
            s = "-" + re.sub(r"[()\s]", "", s)
        v = float(s)
        return None if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return None


def _worker_exec(code, q):
    buf = io.StringIO()
    try:
        import builtins as real_builtins
        safe_env = {k: getattr(real_builtins, k) for k in dir(real_builtins)}

        import json as _json
        import math as _math
        safe_env.update({"json": _json, "math": _math})

        clean_code = code.replace("\xa0", " ")

        with contextlib.redirect_stdout(buf):
            exec(clean_code, safe_env)

        q.put({"stdout": buf.getvalue(), "error": ""})
    except Exception as e:
        q.put({"stdout": buf.getvalue(), "error": repr(e)})


def exec_with_timeout(code, timeout_s=5.0):
    q = mp.Queue()
    p = mp.Process(target=_worker_exec, args=(code, q), daemon=True)
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(1)
        return "", "Timeout", "timeout"

    res = q.get() if not q.empty() else {"stdout": "", "error": "Worker error"}
    return res["stdout"], res["error"], "exec_error" if res["error"] else "ok"


def extract_generated_codes(row):
    """
    New format:
      "generated_codes": [code1, code2, ...]
    Old format:
      "generated_code": "..."
    Always return a list.
    """
    if "generated_codes" in row and isinstance(row["generated_codes"], list):
        return row["generated_codes"]
    if "generated_code" in row:
        return [row.get("generated_code", "")]
    return []


def parse_exec_output(stdout):
    """
    Parse the last non-empty stdout line.
    Prefer JSON {"ans": ..., "scale": ...}.
    Fall back to numeric or raw last line.
    """
    last_line = None
    pred_answer = None
    pred_scale = ""

    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1]

    if last_line is None:
        return last_line, pred_answer, pred_scale

    try:
        parsed_out = json.loads(last_line)
        pred_answer = parsed_out.get("ans")
        pred_scale = parsed_out.get("scale","")
        return last_line, pred_answer, pred_scale
    except Exception:
        pred_answer = to_float_maybe(last_line)
        if pred_answer is None:
            pred_answer = last_line
        return last_line, pred_answer, pred_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--in_jsonl", type=str, help="Path to generated-code JSONL")
    ap.add_argument("--out_jsonl", type=str, help="Path to save run JSONL")
    args = ap.parse_args()

    base_dir = Path("./outputs")
    in_path = Path(args.in_jsonl) if args.in_jsonl else base_dir / f"teacher_codegen_{args.split}.jsonl"
    out_path = Path(args.out_jsonl) if args.out_jsonl else base_dir / f"teacher_codegen_{args.split}_results.jsonl"

    if not in_path.exists():
        print(f"Error: Input file {in_path} not found!")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    row_count = 0
    cand_total = 0
    cand_ok = 0
    cand_err = 0

    print(f"Reading from: {in_path}")

    for line in in_path.open(encoding="utf-8"):
        if not line.strip():
            continue

        row = json.loads(line)
        row_count += 1

        if row_count % 10 == 0:
            print(f"[Progress] processed {row_count} rows...")

        codes = extract_generated_codes(row)

        # Keep all original fields except the list form; expand to *_1, *_2, ...
        rec = {k: v for k, v in row.items() if k not in {"generated_codes", "generated_code"}}
        rec["num_generated_codes"] = len(codes)

        if not codes:
            rec["run_summary_status"] = "missing_code"
            rows.append(rec)
            continue

        per_row_ok = 0
        per_row_err = 0

        for i, code in enumerate(codes, start=1):
            rec[f"generated_code_{i}"] = code

            if not isinstance(code, str) or not code.strip():
                rec[f"run_status_{i}"] = "missing_code"
                rec[f"stdout_last_{i}"] = None
                rec[f"pred_answer_{i}"] = None
                rec[f"pred_scale_{i}"] = None
                rec[f"exec_error_{i}"] = "Missing generated code"
                rec[f"exec_stdout_{i}"] = None
                cand_err += 1
                per_row_err += 1
                cand_total += 1
                continue

            stdout, error, status = exec_with_timeout(code)
            last_line, pred_answer, pred_scale = parse_exec_output(stdout)

            rec[f"run_status_{i}"] = status
            rec[f"stdout_last_{i}"] = last_line
            rec[f"pred_answer_{i}"] = pred_answer
            rec[f"pred_scale_{i}"] = pred_scale
            rec[f"exec_error_{i}"] = error or None
            rec[f"exec_stdout_{i}"] = stdout if status != "ok" else None

            cand_total += 1
            if status == "ok":
                cand_ok += 1
                per_row_ok += 1
            else:
                cand_err += 1
                per_row_err += 1

        rec["run_ok_count"] = per_row_ok
        rec["run_error_count"] = per_row_err
        rec["run_summary_status"] = "ok" if per_row_ok > 0 else "all_failed"

        rows.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Finished rows: {row_count}")
    print(f"Finished candidate executions: {cand_total} (ok: {cand_ok}, errors: {cand_err})")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()