import os
import torch
import sys
import json
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.operators import result_compute, normalize_exp
from func_timeout import func_timeout
import argparse

def read_prediction(file_path):
    data_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            question_id = item['question_id']
            text = item['text']
            data_dict[question_id] = text
    return data_dict

def read_geometry_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def compute_exp_result_topk(pred_file, gt_file, k_num = 3):
    test_preds = read_prediction(pred_file)
    geometry_data = read_geometry_data(gt_file)
    ans_num = eq_num = 0

    for question_id in test_preds.keys():
        geom_entry = next((item for item in geometry_data if item['data_id'] == question_id), None)
        if geom_entry is None:
            assert "no entry found for the question_id"

        num_list = geom_entry['var_list']
        tgt = geom_entry['conversations'][1]['value']
        tgt_result = float(geom_entry['answer'])
        is_ans_same = is_eq_same = False

        pred_list = test_preds[question_id]  # Correct key access
        tgt = [value for value in tgt.split()]

        for j in range(k_num): # top-n
            pred = [value for value in pred_list[j].split()]
            try:
                pred = normalize_exp(pred)
                pred_result = float(func_timeout(2.0, result_compute, \
                        kwargs=dict(num_all_list=num_list, exp_tokens=pred)))
                if pred == tgt:
                    is_ans_same = True
                    is_eq_same = True
                    break
                if abs(pred_result-tgt_result)<5e-3: 
                    is_ans_same = True
                    if len(pred)==len(tgt):
                        is_eq_same = True
                        break
            except:
                pass
        
        if is_ans_same: ans_num +=1
        if is_eq_same: eq_num +=1

    return ans_num / len(test_preds), eq_num / len(test_preds)


def compute_exp_result_choice(pred_file, gt_file):
    test_preds = read_prediction(pred_file)
    geometry_data = read_geometry_data(gt_file)
    ans_num = eq_num = 0
    

    for question_id in test_preds.keys():
        geom_entry = next((item for item in geometry_data if item['data_id'] == question_id), None)
        if geom_entry is None:
            assert "no entry found for the question_id"

        num_list = geom_entry['var_list']
        tgt = geom_entry['conversations'][1]['value']
        choices = geom_entry['choices']
        tgt_result = float(geom_entry['answer'])
        is_find_ans = False

        pred_list = test_preds[question_id]  # Correct key access
        tgt = [value for value in tgt.split()]

        for j in range(len(pred_list)): # pred candi id 
            pred = [value for value in pred_list[j].split()]
            try:
                pred = normalize_exp(pred)
                pred_result = float(func_timeout(2.0, result_compute, \
                                        kwargs=dict(num_all_list=num_list, exp_tokens=pred)))
                if pred == tgt:
                    ans_num += 1
                    eq_num += 1
                    is_find_ans = True
                    break
                for item in choices:
                    if abs(pred_result-item)<5e-2: 
                        is_find_ans = True
                if  is_find_ans and abs(pred_result-tgt_result)<5e-3:
                    ans_num +=1
                    if len(pred)==len(tgt):
                        eq_num += 1
                if  is_find_ans: break
            except:
                pass
        
        if not is_find_ans:
            pred_result = random.choice(choices)
            if  abs(pred_result-tgt_result)<5e-2:
                ans_num +=1

    return ans_num / len(test_preds), eq_num / len(test_preds)


def compute_exp_result_comp(pred_file, gt_file):
    test_preds = read_prediction(pred_file)
    geometry_data = read_geometry_data(gt_file)
    ans_num = eq_num = 0

    for question_id in test_preds.keys():
        geom_entry = next((item for item in geometry_data if item['data_id'] == question_id), None)
        if geom_entry is None:
            assert "no entry found for the question_id"

        num_list = geom_entry['var_list']
        tgt = geom_entry['conversations'][1]['value']
        tgt_result = float(geom_entry['answer'])
        is_ans_same = is_eq_same = False

        pred_list = test_preds[question_id]  # Correct key access
        tgt = [value for value in tgt.split()]

        for j in range(len(pred_list)): # pred candi id 
            pred = [value for value in pred_list[j].split()]
            try:
                pred = normalize_exp(pred)
                pred_result = float(func_timeout(2.0, result_compute, \
                        kwargs=dict(num_all_list=num_list, exp_tokens=pred)))
                if pred == tgt:
                    is_ans_same = True
                    is_eq_same = True
                    break
                if abs(pred_result-tgt_result)<5e-3:
                    is_ans_same = True
                    if len(pred)==len(tgt):
                        is_eq_same = True
                break
            except:
                pass
        
        if is_ans_same: ans_num +=1
        if is_eq_same: eq_num +=1

    return ans_num / len(test_preds), eq_num / len(test_preds)

def main():
    parser = argparse.ArgumentParser(description='Compute expression result comparison.')
    parser.add_argument('--pred_file', type=str, default="", help='Path to the prediction file')
    parser.add_argument(
        '--gt_file', 
        type=str, 
        choices=["./data/pgps9k/test.json", "./data/geometry3k/test.json"],  
        required=True,  
        help='Path to the ground truth file'
    )

    args = parser.parse_args()

    ans_accuracy_comp, _ = compute_exp_result_comp(args.pred_file, args.gt_file)
    ans_accuracy_choice, _ = compute_exp_result_choice(args.pred_file, args.gt_file)
    ans_accuracy_topk, _ = compute_exp_result_topk(args.pred_file, args.gt_file)
    print(f'Completion Answer Accuracy: {ans_accuracy_comp:.4f}')
    print(f'Choice Answer Accuracy: {ans_accuracy_choice:.4f}')
    print(f'Top-K Answer Accuracy: {ans_accuracy_topk:.4f}')
    return ans_accuracy_comp, ans_accuracy_choice, ans_accuracy_topk
    # return 


if __name__ == '__main__':
    main()