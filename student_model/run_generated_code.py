import json, re, math, argparse, io, contextlib, builtins
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
        
        import json, math
        safe_env.update({"json": json, "math": math})

        clean_code = code.replace('\xa0', ' ')

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





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--in_jsonl", type=str, help="")
    ap.add_argument("--out_jsonl", type=str, help="")
    args = ap.parse_args()
    
    base_dir = Path("student_model/outputs")
    in_path = Path(args.in_jsonl) if args.in_jsonl else base_dir / f"student_codegen_{args.split}.jsonl"
    out_path = Path(args.out_jsonl) if args.out_jsonl else base_dir / f"student_codegen_{args.split}_results.jsonl"

    if not in_path.exists():
        print(f"Error: Input file {in_path} not found!")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    ran = ok = err = 0

    print(f"Reading from: {in_path}")
    for line in in_path.open(encoding="utf-8"):
        if not line.strip(): continue
        row = json.loads(line)
        code = row.get("generated_code", "")
        
        if row.get("status") != "ok" or not code.strip():
            row["run_status"] = "missing_code"
            rows.append(row)
            err += 1
            continue

        stdout, error, status = exec_with_timeout(code)
        
        last_line = None
        pred = None
        try:
            lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
            if lines:
                last_line = lines[-1]
                parsed_out = json.loads(last_line)
                pred = parsed_out.get("ans")
                row["pred_scale"] = parsed_out.get("scale")
        except:
            pred = to_float_maybe(last_line) if last_line else None

        row.update({
            "run_status": status,
            "stdout_last": last_line,
            "pred_answer": pred if pred is not None else last_line,
            "exec_error": error or None,
            "exec_stdout": stdout if status != "ok" else None
        })
        rows.append(row)
        
        ran += 1
        if status == "ok":
            ok += 1
        else:
            err += 1

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows: 
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Finished: processed {ran} executions (ok: {ok}, errors: {err}).")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()