import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def cost(in_tokens, out_tokens, in_rate, out_rate):
    return (in_tokens / 1e6) * in_rate + (out_tokens / 1e6) * out_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/token_lengths_all.csv")
    ap.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    ap.add_argument("--in_rate", type=float, default=0.38)    # $ / 1M input tokens
    ap.add_argument("--out_rate", type=float, default=0.40)   # $ / 1M output tokens
    ap.add_argument("--max_new", type=int, nargs="+", default=[500])
    ap.add_argument("--mult", type=float, default=1.0)        # e.g., 3 for no/one/few-shot
    ap.add_argument("--n", type=int, default=0)               # sample N rows (0 = all)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.csv))
    if args.split != "all":
        df = df[df["split"] == args.split]
    if args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=42)

    in_tokens = df["input_tokens"].to_numpy(dtype=np.int64)
    in_sum = int(in_tokens.sum()) * args.mult
    n = int(len(df) * args.mult)

    print(f"rows={len(df)} (effective runs={n}), input_tokens_total={in_sum:,}")
    print(f"pricing: input=${args.in_rate}/1M, output=${args.out_rate}/1M, mult={args.mult}, split={args.split}")

    for m in args.max_new:
        out_sum = int(len(df) * m) * args.mult
        total = cost(in_sum, out_sum, args.in_rate, args.out_rate)
        per_call = cost(in_sum / args.mult / len(df), m, args.in_rate, args.out_rate)  # avg per original row
        print(f"max_new={m:>4} | output_tokens_total={out_sum:,} | total={total:.2f} | avg_per_example={per_call:.6f}")

if __name__ == "__main__":
    main()