#!/usr/bin/python
import argparse #Python 标准库：用来解析命令行参数（--gold_path 这种）。
import json #标准库：读写 JSON 文件。
from tatqa_metric import * #import * 会导入模块里所有不以下划线开头的名字
from typing import Any, Dict, Tuple #只是类型标注用的（不影响运行逻辑）。

# predcition answer 就是个dictonary，question_id: [ans,scale]
# gold answers 感觉就是那个list【ctx】
# em_and_f1(
#  ground_truth = qa,          # 单个 question 的 dict（gold 里的那条）
#  prediction   = pred_answer, # 这一题的预测答案（不是 uid->... 的大 dict）
#  pred_scale   = pred_scale   # 这一题的预测 scale
#)
def evaluate_json(golden_answers: Dict[str, Any], predicted_answers: Dict[str, Any]) -> Tuple[float, float]:
    #对，evaluate_json() 这个函数在驱动打分流程：遍历 gold 里的每道题 → 找到对应 pred → 交给 TaTQAEmAndF1 去累计 → 最后取 overall 指标
    em_and_f1 = TaTQAEmAndF1()
    #不是“算一题就返回分数”的纯函数
    # 而是“一题一题喂进去 → 它内部累计统计 → 最后一次性吐出 overall / 细分指标”的统计器
    for qas in golden_answers:
        for qa in qas["questions"]:
            query_id = qa["uid"]
            pred_answer, pred_scale = None, None
            if query_id in predicted_answers:
                pred_answer, pred_scale = predicted_answers[query_id]
            em_and_f1(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)

    global_em, global_f1, global_scale, global_op = em_and_f1.get_overall_metric()
    print("----")
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("Scale score {0:.2f}".format(global_scale * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")

    detail_raw = em_and_f1.get_raw_pivot_table()
    print("---- raw detail ---")
    print(detail_raw)
    detail_em, detail_f1 = em_and_f1.get_detail_metric()
    print("---- em detail ---")
    print(detail_em)
    print("---- f1 detail ---")
    print(detail_f1)



# 读good example 
def evaluate_prediction_file(gold_path: str,
                             pred_path: str):

    golden_answers = json.load(open(gold_path, encoding='utf-8'))
    predicted_answers = json.load(open(pred_path, encoding='utf-8'))
    evaluate_json(golden_answers, predicted_answers)


if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='evaluation on TAT-QA dataset')
    parser.add_argument("--gold_path",
                        type=str,
                        required=True,
                        default="tatqa_dataset_test_gold.json",
                        help='The path of the gold file')
    parser.add_argument("--pred_path",
                        type=str,
                        required=True,
                        default="sample_predictions.json",
                        help='The path of the prediction file')

    args = parser.parse_args()
    evaluate_prediction_file(args.gold_path, args.pred_path)
