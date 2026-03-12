import json
import argparse


def load_qids(path):
    qids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                qids.append(str(row["qid"]))
    return qids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round2_path", type=str, required=True)
    parser.add_argument("--rest_path", type=str, required=True)
    args = parser.parse_args()

    round2_qids = load_qids(args.round2_path)
    rest_qids = load_qids(args.rest_path)

    round2_set = set(round2_qids)
    rest_set = set(rest_qids)

    overlap = round2_set & rest_set
    union = round2_set | rest_set

    print(f"round2 samples: {len(round2_qids)}")
    print(f"rest samples: {len(rest_qids)}")
    print(f"round2 unique qids: {len(round2_set)}")
    print(f"rest unique qids: {len(rest_set)}")
    print(f"overlap qids: {len(overlap)}")
    print(f"union unique qids: {len(union)}")

    if overlap:
        print("\nOverlapping qids (first 20):")
        for qid in list(sorted(overlap))[:20]:
            print(qid)
    else:
        print("\nNo overlap between the two files.")


if __name__ == "__main__":
    main()