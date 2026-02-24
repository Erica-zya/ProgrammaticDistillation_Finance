import json
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# Stats utilities for the raw / filtered TAT-QA dataset (context-level JSON)
#
# Each JSON file is a list of "contexts". Each context contains:
#   - table: { "uid": str, "table": List[List[str]] }
#   - paragraphs: List[ { "uid": str, "order": int, "text": str } ]
#   - questions: List[ question objects ... ]
#
# Goal:
#   1) Load a JSON file (full or filtered)
#   2) Compute dataset-level stats:
#        - #contexts, #questions
#        - answer_type distribution
#        - answer_from distribution
#        - req_comparison distribution
#        - scale distribution
#        - derivation empty/non-empty (for full data)
#        - cross: answer_type x derivation_empty (for full data)
#   3) Save a JSON summary
#.  4) Generate plots: 
#    - Bar: answer_type distribution (Train/Dev/Test, FULL vs FILTERED)
#    - Bar: answer_from/source distribution (Train/Dev/Test, FULL vs FILTERED)
#    - Heatmap: answer_type × answer_from (4×3, %, Train/Dev/Test, FULL vs FILTERED)
#    - Bar: derivation_empty distribution (empty vs non-empty, Train/Dev/Test, FULL vs FILTERED)
#    - Bar: req_comparison distribution (Train/Dev/Test, FULL vs FILTERED)
# ============================================================

def load_json(json_path: str):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data

def compute_stats(data, split_name="Split"):
    stats = {}
    stats["n_contexts"] = len(data)
    
    answer_type = {}
    answer_from = {}
    req_comparison = {True: 0, False: 0}
    
    n_questions = 0
    for ctx in data:
        for q in ctx['questions']:
            n_questions += 1
            # Counts
            at = q['answer_type']
            answer_type[at] = answer_type.get(at, 0) + 1
            af = q['answer_from']
            answer_from[af] = answer_from.get(af, 0) + 1
            req_comparison[q['req_comparison']] += 1

    stats['n_questions'] = n_questions
    
    # calculates percentages
    stats['percentages'] = {
        'answer_type': {k: (v / n_questions) * 100 for k, v in answer_type.items()},
        'answer_from': {k: (v / n_questions) * 100 for k, v in answer_from.items()},
        'req_comparison': {k: (v / n_questions) * 100 for k, v in req_comparison.items()}
    }
    
    stats['answer_type'] = answer_type
    stats['answer_from'] = answer_from
    stats['req_comparison'] = req_comparison

    return stats

def print_comprehensive_report(full_stats, filt_train, filt_dev, filt_test, label):
    print(f"\n{'='*20} {label.upper()} DISTRIBUTION (%) {'='*20}")
    print(f"{'Category':<15} | {'Full Train':<10} | {'Filt. Train':<10} | {'Filt. Dev':<10} | {'Filt. Test':<10}")
    print("-" * 75)
    
    # Get all unique keys
    all_keys = set(full_stats['percentages'][label].keys()) | \
               set(filt_train['percentages'][label].keys())
    
    for key in sorted(all_keys):
        f_tr  = full_stats['percentages'][label].get(key, 0)
        fi_tr = filt_train['percentages'][label].get(key, 0)
        fi_dv = filt_dev['percentages'][label].get(key, 0)
        fi_ts = filt_test['percentages'][label].get(key, 0)
        
        print(f"{str(key):<15} | {f_tr:>9.1f}% | {fi_tr:>9.1f}% | {fi_dv:>9.1f}% | {fi_ts:>9.1f}%")





def plot_bar_grouped_3splits(train_dict, dev_dict, test_dict, title,x_lable, save_path=None):
    order=list(train_dict.keys())
    train_counts = [train_dict[k] for k in order]
    dev_counts   = [dev_dict[k] for k in order]
    test_counts  = [test_dict[k] for k in order]

    x = list(range(len(order)))
    width = 0.25

    plt.figure()
    plt.bar([i - width for i in x], train_counts, width=width, label="train")
    plt.bar(x,                    dev_counts,   width=width, label="dev")
    plt.bar([i + width for i in x], test_counts, width=width, label="test")

    plt.xticks(x, order, rotation=15)
    plt.xlabel(x_lable)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
    #plt.show()
    plt.close()


########## heatmap
def compute_type_from_table(data, type_order, from_order):
    # 4x3 count table: rows=answer_type, cols=answer_from
    table = [[0 for _ in from_order] for _ in type_order]
    total = 0

    for ctx in data:
        for q in ctx.get("questions", []):
            at = q.get("answer_type", "UNKNOWN")
            af = q.get("answer_from", "UNKNOWN")
            if at in type_order and af in from_order:
                i = type_order.index(at)
                j = from_order.index(af)
                table[i][j] += 1
                total += 1
    return table, total


def plot_heatmap(count_table, title, type_order, from_order,
                        normalize=False, save_path=None):
    # normalize=True will display percentages instead of counts
    if normalize:
        total = sum(sum(row) for row in count_table)
        display = [[(v / total * 100.0) if total > 0 else 0.0 for v in row] for row in count_table]
    else:
        display = count_table

    plt.figure()
    plt.imshow(display, aspect="auto")
    plt.title(title)
    plt.xlabel("Answer source")
    plt.ylabel("Answer type")
    plt.xticks(range(len(from_order)), from_order, rotation=15)
    plt.yticks(range(len(type_order)), type_order)

    # annotate each cell
    for i in range(len(type_order)):
        for j in range(len(from_order)):
            v = display[i][j]
            if normalize:
                text = f"{v:.1f}%"
            else:
                text = str(v)
            plt.text(j, i, text, ha="center", va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    #plt.show()
    plt.close()
################

def rename_derivation_dict(d):
    # True=empty, False=non-empty
    return {
        "empty": d.get(True, 0),
        "non-empty": d.get(False, 0),
    }

def rename_bool_dict(d, true_label, false_label):
    return {
        true_label: d.get(True, 0),
        false_label: d.get(False, 0),
    }



if __name__ == "__main__":
    # 1. Setup paths (resolve everything relative to the project root)
    project_root = Path(__file__).resolve().parent.parent
    save_path = project_root / "plots"
    save_path.mkdir(parents=True, exist_ok=True)

    # Define paths for the comprehensive report (raw and filtered JSON)
    paths = {
        "full_train":   project_root / "dataset_raw" / "tatqa_dataset_train.json",
        "full_dev":     project_root / "dataset_raw" / "tatqa_dataset_dev.json",
        "full_test":    project_root / "dataset_raw" / "tatqa_dataset_test_gold.json",
        "filt_train":   project_root / "dataset_filtered" / "tatqa_dataset_train_filtered.json",
        "filt_dev":     project_root / "dataset_filtered" / "tatqa_dataset_dev_filtered.json",
        "filt_test":    project_root / "dataset_filtered" / "tatqa_dataset_test_gold_filtered.json",
    }

    # 2. Compute Stats for all splits (this replaces the individual compute_stats calls)
    all_stats = {}
    for name, path in paths.items():
        data = load_json(path)
        all_stats[name] = compute_stats(data, name)

    # 3. Generate the Plots (using the all_stats dictionary)
    # Answer type plots
    plot_bar_grouped_3splits(all_stats['full_train']['answer_type'], all_stats['full_dev']['answer_type'], all_stats['full_test']['answer_type'], 
                             "Answer Type Distribution (Full)", x_lable="Type", save_path=str(save_path/"answer_type_distribution_full.png")) 
    plot_bar_grouped_3splits(all_stats['filt_train']['answer_type'], all_stats['filt_dev']['answer_type'], all_stats['filt_test']['answer_type'], 
                             "Answer Type Distribution (Filtered)", x_lable="Type", save_path=str(save_path/"answer_type_distribution_filtered.png"))    

    # 4. Print the Consistency Report Table
    print("\n")
    print("DATASET TRANSFORMATION & CONSISTENCY REPORT (PERCENTAGES)")
    
    # This specifically compares Full Train vs Filtered Splits
    for category in ['answer_type', 'answer_from', 'req_comparison']:
        print_comprehensive_report(
            all_stats["full_train"], 
            all_stats["filt_train"], 
            all_stats["filt_dev"], 
            all_stats["filt_test"], 
            category
        )

    # 5. Volume Summary
    print(f"\n{'='*20} VOLUME SUMMARY {'='*20}")
    n_full = all_stats['full_train']['n_questions']
    n_filt = all_stats['filt_train']['n_questions']
    print(f"Original Train questions: {n_full}")
    print(f"Filtered Train questions: {n_filt}")
    print(f"Retention Rate: {(n_filt/n_full)*100:.1f}%")
    


                         




    #plot_bar_single(full_stats['answer_type'], "Answer type distribution", save_path=None)
    #filtered_stats = compute_stats(filtered_data)

    # Print
   #pretty_print_stats("FULL", full_stats)
   # pretty_print_stats("FILTERED (derivation non-empty)", filtered_stats)
    #compare_full_vs_filtered(full_stats, filtered_stats)