import argparse
import os
import json
from utils.utils import *

def get_args():
    parser = argparse.ArgumentParser(description="Basic test for the TimeBlind benchmark")
    parser.add_argument('--folder_path', metavar='E', type=str, default='/home/ec2-user/zhou/VLM_KD/result/LLM_judge',
                        help='Path to result folder path.')
    parser.add_argument('--save_path', '-sp', metavar='E', type=str, default='/home/ec2-user/zhou/VLM_KD/result/comparison_result.png')
    return parser.parse_args()


def main():
    args = get_args()
    files_path = os.listdir(args.folder_path)
    correct_data = []
    label_map = ['Qwen3VL-8B', 'Qwen3VL-30B']
    for file_name in files_path:
        file_path = os.path.join(args.folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            test_data = [json.loads(line) for line in f if line.strip()]
        json_result = []
        for i, item in enumerate(test_data):
            if i == 200:
                break
            json_result.append(1 if item['answer'] == item['predict_answer'] else 0)
        correct_data.append(json_result)
    compare_model_results(answer_maps=correct_data, model_names=label_map, save_path=args.save_path)



if __name__ == '__main__':
    main()