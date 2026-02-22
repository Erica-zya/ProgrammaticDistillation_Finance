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
#      - answer type distribution
#.     - answer source distributiom
#      - heatmap: answer type x answer source
# ============================================================

def load_json(json_path: str):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data

def compute_stats(data):
    stats = {}
    stats["n_contexts"] = len(data)
    
    answer_type = {}
    answer_from = {}
    req_comparison = {True: 0, False: 0}
    scale = {}
    derivation_empty = {True: 0, False: 0}

    n_quesions=0
    for ctx in data:
        questions=ctx['questions']
        for q in questions:
            n_quesions+=1
            ans_t=q['answer_type']
            answer_type[ans_t]=answer_type.get(ans_t,0)+1
            ans_f=q['answer_from']
            answer_from[ans_f]=answer_from.get(ans_f,0)+1
            r_compare=q['req_comparison']
            req_comparison[r_compare]+=1
            #d=q['derivation']
            d = q.get("derivation", "")
            if isinstance(d, str) and d.strip() != "":
                derivation_empty[False]+=1
            else:
                derivation_empty[True]+=1
            s = q.get("scale", "")
            if not isinstance(s, str):
                s = str(s)
            scale[s] = scale.get(s, 0) + 1
    
    stats['n_questions']=n_quesions
    stats['answer_type']=answer_type
    stats['answer_from']=answer_from
    stats['req_comparison']=req_comparison
    stats["derivation_empty"]=derivation_empty
    stats["scale"]=scale
    
    return stats





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

if __name__ == "__main__":
    # --- Change these two lines to switch datasets ---
    full_train = "../dataset_raw/tatqa_dataset_train.json"
    filtered_train = "../dataset_filtered/tatqa_dataset_train_filtered.json"
    full_dev = "../dataset_raw/tatqa_dataset_dev.json"
    filtered_dev = "../dataset_filtered/tatqa_dataset_dev_filtered.json"
    full_test = "../dataset_raw/tatqa_dataset_test_gold.json"
    filtered_test = "../dataset_filtered/tatqa_dataset_test_gold_filtered.json"

    # Load
    full_train_data = load_json(full_train)
    filtered_train_data = load_json(filtered_train)
    full_dev_data = load_json(full_dev)
    filtered_dev_data = load_json(filtered_dev)
    full_test_data = load_json(full_test)
    filtered_test_data = load_json(filtered_test)

    # Compute
    full_train_stats = compute_stats(full_train_data)
    filtered_train_stats = compute_stats(filtered_train_data)
    full_dev_stats = compute_stats(full_dev_data)
    filtered_dev_stats = compute_stats(filtered_dev_data)
    full_test_stats = compute_stats(full_test_data)
    filtered_test_stats = compute_stats(filtered_test_data)

    # plot 
    save_path='../plots/'
    # Answer type
    plot_bar_grouped_3splits(full_train_stats['answer_type'], full_dev_stats['answer_type'], full_test_stats['answer_type'], "Answer Type Distribution", x_lable="Type",save_path=save_path+"answer_type_distribution_full.png") 
    plot_bar_grouped_3splits(filtered_train_stats['answer_type'], filtered_dev_stats['answer_type'], filtered_test_stats['answer_type'], "Answer Type Distribution", x_lable="Type", save_path=save_path+"answer_type_distribution_filtered.png")    
    # Answer from
    plot_bar_grouped_3splits(full_train_stats['answer_from'], full_dev_stats['answer_from'], full_test_stats['answer_from'], "Answer Source Distribution", x_lable="Answer Source", save_path=save_path+"answer_from_distribution_full.png") 
    plot_bar_grouped_3splits(filtered_train_stats['answer_from'], filtered_dev_stats['answer_from'], filtered_test_stats['answer_from'], "Answer Source Distribution", x_lable="Answer Source", save_path=save_path+"answer_from_distribution_filtered.png")    

    # heatmap
    TYPE_ORDER = ["span", "multi-span", "arithmetic", "count"]
    FROM_ORDER = ["table", "text", "table-text"]
    # full train
    table_train_full, total_train_full=compute_type_from_table(full_train_data, type_order=TYPE_ORDER, from_order=FROM_ORDER)
    plot_heatmap(table_train_full, f"Answer Type × Answer Source (Train %)  (N={total_train_full})", TYPE_ORDER, FROM_ORDER , normalize=True, save_path=save_path+"heatmap_type_x_source_train_full_pct.png")
    # full dev train
    table_dev_full, total_dev_full=compute_type_from_table(full_dev_data, type_order=TYPE_ORDER, from_order=FROM_ORDER)
    plot_heatmap(table_dev_full, f"Answer Type × Answer Source (Dev %)  (N={total_dev_full})", TYPE_ORDER, FROM_ORDER , normalize=True, save_path=save_path+"heatmap_type_x_source_dev_full_pct.png")
    # full test train
    table_test_full, total_test_full=compute_type_from_table(full_test_data, type_order=TYPE_ORDER, from_order=FROM_ORDER)
    plot_heatmap(table_test_full, f"Answer Type × Answer Source (Test %)  (N={total_test_full})", TYPE_ORDER, FROM_ORDER , normalize=True, save_path=save_path+"heatmap_type_x_source_test_full_pct.png")

     # filtered train
    table_train_filtered, total_train_filtered=compute_type_from_table(filtered_train_data, type_order=TYPE_ORDER, from_order=FROM_ORDER)
    plot_heatmap(table_train_filtered, f"Answer Type × Answer Source (Train %)  (N={total_train_filtered})", TYPE_ORDER, FROM_ORDER , normalize=True, save_path=save_path+"heatmap_type_x_source_train_filtered_pct.png")
    # full dev train
    table_dev_filtered, total_dev_filtered=compute_type_from_table(filtered_dev_data, type_order=TYPE_ORDER, from_order=FROM_ORDER)
    plot_heatmap(table_dev_filtered, f"Answer Type × Answer Source (Dev %)  (N={total_dev_filtered})", TYPE_ORDER, FROM_ORDER , normalize=True, save_path=save_path+"heatmap_type_x_source_dev_filtered_pct.png")
    # full test train
    table_test_filtered, total_test_filtered=compute_type_from_table(filtered_test_data, type_order=TYPE_ORDER, from_order=FROM_ORDER)
    plot_heatmap(table_test_filtered, f"Answer Type × Answer Source (Test %)  (N={total_test_filtered})", TYPE_ORDER, FROM_ORDER , normalize=True, save_path=save_path+"heatmap_type_x_source_test_filtered_pct.png")

    


                         




    #plot_bar_single(full_stats['answer_type'], "Answer type distribution", save_path=None)
    #filtered_stats = compute_stats(filtered_data)

    # Print
   #pretty_print_stats("FULL", full_stats)
   # pretty_print_stats("FILTERED (derivation non-empty)", filtered_stats)
    #compare_full_vs_filtered(full_stats, filtered_stats)