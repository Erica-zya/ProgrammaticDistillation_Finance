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
    ap.add_argument("--in_jsonl", default="teacher_model/outputs/teacher_codegen_train.jsonl")
    ap.add_argument("--out_jsonl", default="teacher_model/outputs/teacher_codegen_train_results.jsonl")
    args = ap.parse_args()
    
    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    rows = []
    ran = ok = err = 0

    for line in in_path.open(encoding="utf-8"):
        row = json.loads(line)
        code = row.get("generated_code", "")
        
        if row.get("status") != "ok" or not code.strip():
            row["run_status"] = "missing_code"
            rows.append(row)
            err += 1
            continue

        stdout, error, status = exec_with_timeout(code)
        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        last_line = lines[-1] if lines else None
        pred = to_float_maybe(last_line)

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
            
    print(f"Finished: processed {ran} executions (ok: {ok}, errors: {err}). Results saved to {out_path.name}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()