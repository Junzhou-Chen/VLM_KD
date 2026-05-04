import os
import argparse
import json
from tqdm import tqdm

from utils.load_intern_vl import InternVL
from utils.utils import *


def load_timeblind(benchmark_path: str):
    file_path = os.path.join(benchmark_path, 'data.jsonl')
    with open(file_path, "r", encoding="utf-8") as f:
        qa_data = [json.loads(line) for line in f if line.strip()]
    for i in range(len(qa_data)):
        qa_data[i]['video_path'] = os.path.join(benchmark_path, os.path.relpath(qa_data[i]['video_path'], start='TimeBlind'))
    return qa_data

def load_message(video_path: str, question: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    return messages

def get_args():
    parser = argparse.ArgumentParser(description="Basic test for the TimeBlind benchmark")
    parser.add_argument('--folder_path', '-m', metavar='E', type=str, default='/home/ec2-user/zhou/VLM_KD/result/Qwen',
                        help='Path of test json file save path')
    parser.add_argument('--benchmark', '-b', metavar='E', type=str, default='/home/ec2-user/zhou/dataset/TimeBlind', help='Path of benchmark')
    parser.add_argument('--model_path', '-mp', metavar='E', type=str, default=None, help='Path of test model')
    parser.add_argument('--save_path', '-sp', metavar='E', type=str, default='/home/ec2-user/zhou/VLM_KD/result')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    model = InternVL(args.model_name)
    file_name = args.model_name + '_timeblind.jsonl'
    save_name = os.path.join(args.save_path, file_name)
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    if os.path.exists(save_name):
        with open(save_name, "r", encoding="utf-8") as f:
            blind_qa = [json.loads(line) for line in f if line.strip()]
    else:
        blind_qa = load_timeblind(args.benchmark)
    for i, item in enumerate(tqdm(blind_qa, desc="Testing", total=len(blind_qa))):
        if "model_output" in blind_qa[i]:
            continue
        message = load_message(video_path=item['video_path'], question=item['question'])
        predict_answer = model.process_message(message=message, max_new_tokens=512)
        blind_qa[i]['model_output'] = predict_answer
        print(predict_answer)
        if i % 200 == 0:
            with open(save_name, "w", encoding="utf-8") as f:
                for item in blind_qa:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(save_name, "w", encoding="utf-8") as f:
        for item in blind_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")




