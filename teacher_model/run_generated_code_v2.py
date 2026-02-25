import json, re, math, argparse, io, contextlib, builtins
import multiprocessing as mp
from pathlib import Path

# Remove `import json` so we don't need __import__ in the sandbox
_IMPORT_JSON_LINE = re.compile(r"^\s*import\s+json\s*(?:#.*)?$", re.MULTILINE)

def sanitize_code(code: str) -> str:
    return _IMPORT_JSON_LINE.sub("", code).strip()

def _worker_exec(code, q):
    buf = io.StringIO()
    try:
        allowed = (
            'abs','min','max','sum','round',
            'int','float','str','len','range','enumerate',
            'list','dict','tuple','set',
            'print'
        )
        safe_env = {k: getattr(builtins, k) for k in allowed}
        safe_env['math'] = math
        safe_env['json'] = json  # allow json.dumps/json.loads without import

        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": safe_env}, {})
        q.put({"stdout": buf.getvalue(), "error": ""})
    except Exception as e:
        q.put({"stdout": buf.getvalue(), "error": repr(e)})

def exec_with_timeout(code, timeout_s=2.0):
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
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--timeout_s", type=float, default=2.0)
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    ran = ok = err = 0

    for line in in_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)

        code = row.get("generated_code", "")
        gold_ans = row.get("gold_answer", row.get("answer"))
        gold_scale = row.get("gold_scale")

        if row.get("status") != "ok" or not code.strip():
            row.update({
                "run_status": "missing_code",
                "stdout_last": None,
                "pred_ans": None,
                "pred_scale": None,
                #"gold_ans": gold_ans,
                #"gold_scale": gold_scale,
            })
            rows.append(row)
            err += 1
            continue

        code2 = sanitize_code(code)
        stdout, error, status = exec_with_timeout(code2, timeout_s=args.timeout_s)

        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        last_line = lines[-1] if lines else None

        pred_ans = None
        pred_scale = None
        if last_line:
            s = last_line.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    pred_ans = obj.get("ans")
                    pred_scale = obj.get("scale")
                except Exception:
                    pred_ans = s
            else:
                pred_ans = s

        row.update({
            "run_status": status,
            "exec_error": error or None,
            "stdout_last": last_line,
            "pred_ans": pred_ans,
            "pred_scale": pred_scale,
            "gold_ans": gold_ans,
            "gold_scale": gold_scale,
            "exec_stdout": stdout if status != "ok" else None,
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

    print(f"Finished: ran {ran} (ok: {ok}, errors: {err}). Saved to {out_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()