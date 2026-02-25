import json
import re
import string
import argparse
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation) - {'.', '-'}
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return str(text).lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def is_match(p, g):
    if p == g: return True
    try:
        pf = float(p)
        gf = float(g)
        if abs(pf - gf) < 1e-4: return True
        if round(pf, 1) == round(gf, 1): return True
        if round(pf, 2) == round(gf, 2): return True
        if round(pf, 3) == round(gf, 3): return True
    except ValueError:
        pass
    return False

def _answer_to_bags(answer):
    raw_spans = answer if isinstance(answer, (list, tuple)) else [answer]
    normalized_spans = [normalize_answer(str(span)) for span in raw_spans]
    token_bags = [set(span.split()) for span in normalized_spans]
    return normalized_spans, token_bags

def _compute_f1(predicted_bag, gold_bag):
    pb = list(predicted_bag)
    gb = list(gold_bag)
    match_count = 0
    used_g = set()
    for p in pb:
        for i, g in enumerate(gb):
            if i not in used_g and is_match(p, g):
                match_count += 1
                used_g.add(i)
                break
    precision = 1.0 if not pb else match_count / float(len(pb))
    recall = 1.0 if not gb else match_count / float(len(gb))
    return (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0

def _align_bags(predicted, gold):
    scores = np.zeros([len(gold), len(predicted)])
    for g_idx, g_item in enumerate(gold):
        for p_idx, p_item in enumerate(predicted):
            scores[g_idx, p_idx] = _compute_f1(p_item, g_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores

def get_metrics(predicted, gold):
    pred_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    
    exact_match = 0.0
    if len(pred_bags[0]) == len(gold_bags[0]):
        all_match = True
        used_g = set()
        for p_span in pred_bags[0]:
            match_found = False
            for i, g_span in enumerate(gold_bags[0]):
                if i not in used_g and is_match(p_span, g_span):
                    match_found = True
                    used_g.add(i)
                    break
            if not match_found:
                all_match = False
                break
        if all_match:
            exact_match = 1.0
            
    f1 = round(np.mean(_align_bags(pred_bags[1], gold_bags[1])), 2)
    return exact_match, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", type=str, required=True)
    parser.add_argument("--gold_json", type=str, required=True)
    args = parser.parse_args()

    predictions = {}
    with open(args.in_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            pred_id = data.get("qid", "") 
            pred_str = data.get("pred_answer", "")
            pred_ans = ""
            
            if pred_str:
                try:
                    pred_dict = json.loads(pred_str)
                    pred_ans = pred_dict.get("ans", "")
                except json.JSONDecodeError:
                    pred_ans = pred_str
                    
            predictions[pred_id] = pred_ans

    details = []
    with open(args.gold_json, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    for item in gold_data:
        for qa in item['questions']:
            uid = qa['uid']
            if uid not in predictions: continue
            
            pred_ans = predictions[uid]
            if not isinstance(pred_ans, list): 
                pred_ans = [str(pred_ans)]
            else: 
                pred_ans = [str(ans) for ans in pred_ans]
                
            gold_ans = qa['answer']
            best_em, best_f1 = get_metrics(pred_ans, gold_ans)

            details.append({
                "uid": uid,
                "question": qa.get("question", ""),
                "answer_type": qa.get("answer_type", "unknown"),
                "answer_from": qa.get("answer_from", "unknown"),
                "gold_answer": gold_ans,
                "pred_answer": pred_ans,
                "em": best_em,
                "f1": best_f1
            })

    if not details:
        print("Error: No matching IDs found between JSONL and Gold JSON.")
        return

    out_dir = os.path.dirname(args.in_jsonl) if os.path.dirname(args.in_jsonl) else "."
    correct_path = os.path.join(out_dir, "correct_predictions.jsonl")
    incorrect_path = os.path.join(out_dir, "incorrect_predictions.jsonl")

    correct_count = 0
    incorrect_count = 0

    with open(correct_path, "w", encoding="utf-8") as f_corr, open(incorrect_path, "w", encoding="utf-8") as f_inc:
        for d in details:
            if d["em"] == 1.0:
                f_corr.write(json.dumps(d, ensure_ascii=False) + "\n")
                correct_count += 1
            else:
                f_inc.write(json.dumps(d, ensure_ascii=False) + "\n")
                incorrect_count += 1

    df = pd.DataFrame(details)
    raw_matrix = df.pivot_table(index='answer_type', columns='answer_from', values='em', aggfunc='count').fillna(0)
    em_matrix = df.pivot_table(index='answer_type', columns='answer_from', values='em', aggfunc='mean').fillna(0)
    f1_matrix = df.pivot_table(index='answer_type', columns='answer_from', values='f1', aggfunc='mean').fillna(0)

    print("\nraw matrix:                em")
    print(raw_matrix.to_string())
    print("\ndetail em:                   em")
    print(em_matrix.to_string())
    print("\ndetail f1:                   f1")
    print(f1_matrix.to_string())
    print(f"\nglobal em:{df['em'].mean()}")
    print(f"\nglobal f1:{df['f1'].mean()}")
    print(f"\nSaved {correct_count} correct outputs to: {correct_path}")
    print(f"Saved {incorrect_count} incorrect outputs to: {incorrect_path}")

if __name__ == '__main__':
    main()