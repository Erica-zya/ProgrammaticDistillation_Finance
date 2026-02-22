import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# =========================
# Config
# =========================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# change to your own path
DATA_DIR = Path(r"D:\Stanford\2026 winter\ProgrammaticDistillation_Finance\dataset_filtered")
SPLITS = {
    "train": "tatqa_dataset_train_filtered",
    "dev": "tatqa_dataset_dev_filtered",
    "test": "tatqa_dataset_test_gold_filtered",
}
INCLUDE_SCALE = True
MAX_NEW_TOKENS_ASSUMED = 500
OUT_DIR = Path("outputs")
TOP_N_LONGEST = 50

PROMPT_TMPL = """You are a careful financial reasoning assistant.
Your task: Given TABLE and PARAGRAPHS, write a Python program that computes the answer.

Constraints:
- Use only the provided context (TABLE + PARAGRAPHS).
- Handle scale/unit correctly (e.g., percent, million, billion, thousand).
- Print the final numerical answer using a print() statement.
- Return ONLY valid Python code.

TABLE:
{table}

PARAGRAPHS:
{paras}

QUESTION:
{question}
"""


# =========================
# IO / Formatting
# =========================
def resolve_json_path(stem: str) -> Path:
    p = DATA_DIR / stem
    if p.is_file():
        return p
    p = DATA_DIR / f"{stem}.json"
    if p.is_file():
        return p
    raise FileNotFoundError(f"Missing: {DATA_DIR / stem} or {DATA_DIR / (stem + '.json')}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def table_text(table_obj) -> str:
    rows = (table_obj or {}).get("table", [])
    return "\n".join(" | ".join(map(str, r)) for r in rows)


def paras_text(paragraphs) -> str:
    ps = sorted(paragraphs or [], key=lambda x: x.get("order", 0))
    return "\n".join(p.get("text", "") for p in ps)


def iter_samples(data, split: str):
    if isinstance(data, dict):
        yield from ((str(k), v) for k, v in data.items())
        return
    if isinstance(data, list):
        for i, s in enumerate(data):
            sid = s.get("doc_id") or s.get("id") or s.get("uid") or s.get("question_id") or f"{split}_{i}"
            yield str(sid), s
        return
    raise TypeError(f"Unsupported dataset type: {type(data)}")


def build_prompt(sample, q, gold: bool) -> str:
    p = PROMPT_TMPL.format(
        table=table_text(sample.get("table")),
        paras=paras_text(sample.get("paragraphs")),
        question=q.get("question", ""),
    )
    if gold and q.get("derivation"):
        p += f"\nGOLD_DERIVATION (teacher-only hint):\n{q['derivation']}\n"
    if INCLUDE_SCALE and q.get("scale"):
        p += f"\nSCALE:\n{q['scale']}\n"
    return p + "\n# Python code:\n"


# =========================
# Core
# =========================
def compute_rows(tokenizer, data, split: str):
    total = len(data) if hasattr(data, "__len__") else None
    rows = []

    for doc_id, sample in tqdm(iter_samples(data, split), desc=f"Tokenizing {split}", total=total):
        for qi, q in enumerate(sample.get("questions", [])):
            n_in = len(tokenizer.encode(build_prompt(sample, q, gold=True)))
            rows.append({
                "split": split,
                "doc_id": doc_id,
                "question_idx": qi,
                "answer_type": q.get("answer_type", ""),
                "scale": q.get("scale", ""),
                "input_tokens": n_in,
                "assumed_output_tokens": MAX_NEW_TOKENS_ASSUMED,
                "assumed_total_tokens": n_in + MAX_NEW_TOKENS_ASSUMED,
                "question": q.get("question", ""),
            })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def pct(a, p): return float(np.percentile(a, p))
    out = []
    for split, g in df.groupby("split", sort=True):
        x = g["input_tokens"].to_numpy()
        t = g["assumed_total_tokens"].to_numpy()
        out.append({
            "split": split,
            "n_examples": int(len(x)),
            "mean": float(x.mean()),
            "p50": pct(x, 50),
            "p90": pct(x, 90),
            "p95": pct(x, 95),
            "p99": pct(x, 99),
            "max": int(x.max()),
            "assumed_output_tokens": int(MAX_NEW_TOKENS_ASSUMED),
            "mean_total_tokens": float(t.mean()),
            "p95_total_tokens": pct(t, 95),
            "max_total_tokens": int(t.max()),
        })
    return pd.DataFrame(out)


# =========================
# Plots
# =========================
def plot_hist(df: pd.DataFrame, path: Path):
    plt.figure()
    for split, g in df.groupby("split", sort=True):
        plt.hist(g["input_tokens"], bins=50, alpha=0.5, label=split)
    plt.xlabel("Input tokens (teacher prompt)")
    plt.ylabel("Count")
    plt.title("Token Length Histogram by Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cdf(df: pd.DataFrame, path: Path):
    plt.figure()
    for split, g in df.groupby("split", sort=True):
        x = np.sort(g["input_tokens"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=split)
    plt.xlabel("Input tokens (teacher prompt)")
    plt.ylabel("CDF")
    plt.title("Token Length CDF by Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_box(df: pd.DataFrame, path: Path):
    plt.figure()
    splits = sorted(df["split"].unique())
    data = [df.loc[df["split"] == s, "input_tokens"].to_numpy() for s in splits]
    plt.boxplot(data, tick_labels=splits, showfliers=False)
    plt.xlabel("Split")
    plt.ylabel("Input tokens (teacher prompt)")
    plt.title("Token Length Boxplot by Split")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    rows = []
    for split, stem in SPLITS.items():
        path = resolve_json_path(stem)
        print(f"\nReading {split}: {path}")
        rows.extend(compute_rows(tok, load_json(path), split))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "token_lengths_all.csv", index=False, encoding="utf-8")
    print(f"\nSaved detail CSV: {OUT_DIR / 'token_lengths_all.csv'}")

    summ = summarize(df)
    summ.to_csv(OUT_DIR / "token_stats_summary.csv", index=False, encoding="utf-8")
    print(f"Saved summary CSV: {OUT_DIR / 'token_stats_summary.csv'}")
    print("\n=== Summary ===")
    print(summ.to_string(index=False))

    top = df.nlargest(TOP_N_LONGEST, "input_tokens")
    top_path = OUT_DIR / "top_longest_examples.jsonl"
    with top_path.open("w", encoding="utf-8") as f:
        for r in top.itertuples(index=False):
            f.write(json.dumps({
                "split": r.split,
                "doc_id": r.doc_id,
                "question_idx": int(r.question_idx),
                "input_tokens": int(r.input_tokens),
                "answer_type": r.answer_type,
                "scale": r.scale,
                "question": r.question,
            }, ensure_ascii=False) + "\n")
    print(f"Saved top-longest JSONL: {top_path}")

    plot_hist(df, OUT_DIR / "token_hist_by_split.png")
    plot_cdf(df, OUT_DIR / "token_cdf_by_split.png")
    plot_box(df, OUT_DIR / "token_box_by_split.png")
    print(f"Saved plots to: {OUT_DIR.resolve()}")

    print(f"\nNote: total token estimates assume output max_new_tokens = {MAX_NEW_TOKENS_ASSUMED}.")


if __name__ == "__main__":
    main()