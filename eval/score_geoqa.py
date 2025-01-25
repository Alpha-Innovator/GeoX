import json
import pickle
import numpy as np
from pathlib import Path
from pprint import pformat
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.eval_equ import Equations
import argparse
SUB_DICT_PATH = "data/unigeo/sub_dataset_dict.pk"

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


class GeometryEvaluator:
    def __init__(self):

        self._equ = Equations()

        self.calculation_acc = AverageMeter()
        self.calculation_no_result = AverageMeter()
        self.proving_acc = AverageMeter()
        self.proving_no_result = AverageMeter()

        self.cal_angle = AverageMeter()
        self.cal_length = AverageMeter()
        self.cal_other = AverageMeter()
        self.prove_parallel = AverageMeter()
        self.prove_triangle = AverageMeter()
        self.prove_quadrilateral = AverageMeter()
        self.prove_congruent = AverageMeter()
        self.prove_similarity = AverageMeter()


        with open(SUB_DICT_PATH, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return data

    def load_pickle_data(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    

    def process_file(self, json_file, pk_file):
        json_data = self.load_data(json_file)
        pk_data = self.load_pickle_data(pk_file)

        problem_form = 'calculation'

        for result in json_data:
            data_id = result["question_id"]
            corresponding_data = next(item for item in pk_data if item["id"] == data_id)


            self.geo_evaluation_calc(result['text'], corresponding_data)


    def geo_evaluation_calc(self, preds, batch):
        source_nums = batch['numbers']
        choice_nums = batch['choice_nums']
        label = batch['label']

        problem_type = self.subset_dict[batch['id']]

        # Assume preds is a list of top-k predictions from beam search
        for pred in preds:
            choice = self.evaluate_calculation(pred, choice_nums, source_nums)
            if choice is not None:
                break

        if choice is None:
            self.calculation_acc.update(0)
            self.calculation_no_result.update(1.0)
        elif choice == label:
            self.calculation_acc.update(1.0)
            self.calculation_no_result.update(0)
        else:
            self.calculation_acc.update(0)
            self.calculation_no_result.update(0)

        flag = 1.0 if choice == label else 0
        if problem_type == 'angle':
            self.cal_angle.update(flag)
        elif problem_type == 'length':
            self.cal_length.update(flag)
        else:
            self.cal_other.update(flag)

    def evaluate_calculation(self, prediction, choice_nums, source_nums):
        hypo = prediction.split()
        try:
            res = self._equ.excuate_equation(hypo, source_nums)
        except:
            res = None

        if res is not None and len(res) > 0:
            for i, choice in enumerate(choice_nums):
                if choice_nums[i] is not None and abs(res[-1] - choice) < 0.001:
                    return i
        return None

    def print_results(self):
        # Titles for the different sections
        headers = ["Metric", "Value"]
        calculation_metrics = [
            ("Accuracy", self.calculation_acc.get_avg() * 100),
            ("No Result", self.calculation_no_result.get_avg() * 100),
            ("Angle", self.cal_angle.get_avg() * 100),
            ("Length", self.cal_length.get_avg() * 100),
            ("Other", self.cal_other.get_avg() * 100)
        ]

        # Determine the maximum length of metric names to ensure uniform table width
        max_len = max(len(metric[0]) for metric in calculation_metrics)

        # Function to create a single formatted table
        def format_table(title, metrics):
            title_line = f" {title} ".center(max_len + 15, "-")
            header_line = f"| {'Metric'.ljust(max_len)} | Value (%) |"
            divider_line = "-" * len(header_line)
            rows = [divider_line, title_line, divider_line, header_line, divider_line]
            for metric, value in metrics:
                row = f"| {metric.ljust(max_len)} | {value:7.2f} |"
                rows.append(row)
            rows.append(divider_line)
            return "\n".join(rows)

        # Print tables with adjusted widths
        calculation_table = format_table("Calculation", calculation_metrics)
        # Ensuring both tables are the same width
        print(calculation_table)

        return self.calculation_acc.get_avg() * 100



def main():
    parser = argparse.ArgumentParser(description="Evaluate geometry results.")

    parser.add_argument('--cal_pred_file', type=str, required=True,
                        help='Path to the calculation results JSONL file.')
    parser.add_argument('--cal_gt_file', type=str, default="./data/geoqa/test.pk",
                        help='Path to the calculation PK file.')
    
    
    args = parser.parse_args()
    evaluator = GeometryEvaluator()

    evaluator.process_file(args.cal_pred_file, args.cal_gt_file)
    evaluator.print_results()

if __name__ == "__main__":
    main()
