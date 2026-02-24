import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter


def load_json(json_path: str):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data


def get_first_N_questions(n:int,data:list):
    questions=[]
    N_q=0
    for ctx in data:
        qs=ctx['questions']
        for q in qs:
            questions.append(q)
            N_q+=1
            if N_q ==n:
                return questions
    return questions # no enough n

def _is_nonempty_str(x):
    return isinstance(x, str) and x.strip() != ""


def summarize_questions(questions):
    """Return counters needed for plots."""
    c_type = Counter()
    c_from = Counter()
    c_deriv_empty = Counter()   # keys: True(empty)/False(non-empty)
    c_req_comp = Counter()      # keys: True/False

    for q in questions:
        at = q.get("answer_type", "UNKNOWN")
        af = q.get("answer_from", "UNKNOWN")
        d  = q.get("derivation", "")
        rc = q["req_comparison"] 

        c_type[str(at)] += 1
        c_from[str(af)] += 1
        c_deriv_empty[not _is_nonempty_str(d)] += 1
        c_req_comp[rc] += 1

    return c_type, c_from, c_deriv_empty, c_req_comp

def plot_bar(counter: Counter, title: str, x_label: str, order=None):
    if order is None:
        keys = [k for k, _ in counter.most_common()]
    else:
        keys = list(order) + [k for k in counter.keys() if k not in set(order)]

    vals = [counter.get(k, 0) for k in keys]

    plt.figure(figsize=(7, 4))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=15, ha="right")
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def compute_heatmap_type_x_from(questions, type_order, from_order):
    # 4x3 counts
    type_idx = {t: i for i, t in enumerate(type_order)}
    from_idx = {f: j for j, f in enumerate(from_order)}
    mat = [[0 for _ in from_order] for _ in type_order]

    dropped = 0
    total_used = 0

    for q in questions:
        at = str(q.get("answer_type", "UNKNOWN"))
        af = str(q.get("answer_from", "UNKNOWN"))
        if at in type_idx and af in from_idx:
            mat[type_idx[at]][from_idx[af]] += 1
            total_used += 1
        else:
            dropped += 1

    return mat, total_used, dropped


def plot_heatmap_type_x_from(questions, title):
    TYPE_ORDER = ["span", "multi-span", "arithmetic", "count"]
    FROM_ORDER = ["table", "text", "table-text"]

    mat, total_used, dropped = compute_heatmap_type_x_from(questions, TYPE_ORDER, FROM_ORDER)

    denom = total_used if total_used > 0 else 1
    mat_pct = [[v / denom * 100.0 for v in row] for row in mat]

    plt.figure(figsize=(7, 4))
    plt.imshow(mat_pct, aspect="auto")
    plt.title(f"{title}")
    plt.xlabel("answer_from")
    plt.ylabel("answer_type")
    plt.xticks(range(len(FROM_ORDER)), FROM_ORDER, rotation=15, ha="right")
    plt.yticks(range(len(TYPE_ORDER)), TYPE_ORDER)

    for i in range(len(TYPE_ORDER)):
        for j in range(len(FROM_ORDER)):
            plt.text(j, i, f"{mat_pct[i][j]:.1f}%", ha="center", va="center")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    full_train = "../dataset_raw/tatqa_dataset_train.json"
    # load data
    full_train_data = load_json(full_train)
    N=50
    first_Nq=get_first_N_questions(N,full_train_data)
    #print(first_Nq)

    print("N questions:", len(first_Nq))

    c_type, c_from, c_deriv_empty, c_req_comp = summarize_questions(first_Nq)
    print("answer_type:", dict(c_type))
    print("answer_from:", dict(c_from))
    print("derivation_empty:", {"empty": c_deriv_empty.get(True, 0), "non-empty": c_deriv_empty.get(False, 0)})
    print("req_comparison:", {"True": c_req_comp.get(True, 0), "False": c_req_comp.get(False, 0)})

    # answer_type
    plot_bar(c_type, f"answer_type distribution (first {N} train questions)", "answer_type")

    plot_bar(c_from, title=f"answer_from distribution (first {N} train questions)",x_label="answer_from")
    plot_heatmap_type_x_from(first_Nq, title=f"answer_type × answer_from (first {N} train questions)")

    deriv_named = Counter({
    "empty": c_deriv_empty.get(True, 0),
    "non-empty": c_deriv_empty.get(False, 0)})
plot_bar(deriv_named, f"derivation empty? (first {N} train questions)", "derivation")
req_named = Counter({
    "requires_comparison": c_req_comp.get(True, 0),
    "no_comparison": c_req_comp.get(False, 0),
})
plot_bar(req_named, f"req_comparison (first {N} train questions)", "req_comparison")