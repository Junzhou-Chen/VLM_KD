import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class Qwen3:
    def __init__(self, model_name: str = "Qwen/Qwen3-14B", device_map: str = 'cuda'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          torch_dtype="auto",
                                                          device_map=device_map,
                                                          )
    def process_message(self, prompt, max_new_tokens: int = 32768):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content, thinking_content

def create_extraction_prompt(model_output: str, question_type: str) -> str:
    """Create a strict extraction prompt based on the question type."""
    if question_type == "yes_no":
        return f"""You must extract the final answer from the model output.

        STRICT OUTPUT FORMAT (REQUIRED):
        - Output exactly one token: yes OR no
        - Use lowercase only.
        - Do NOT output anything else (no words, no punctuation, no quotes, no explanation).
        - If you are uncertain, choose the most likely option.
        
        Model output:
        {model_output}
        """
    elif question_type == "multiple_choice":
        return f"""You must extract the final answer from the model output.
            
        STRICT OUTPUT FORMAT (REQUIRED):
        - Output exactly one character: A OR B
        - Use uppercase only.
        - Do NOT output anything else (no words, no punctuation, no quotes, no explanation).
        - If you are uncertain, choose the most likely option.
        
        Model output:
        {model_output}
        """
    else:
        raise ValueError(f"Unknown question type: {question_type}")


def get_args():
    parser = argparse.ArgumentParser(description="Basic test for the TimeBlind benchmark")
    parser.add_argument('--result_file', metavar='E', type=str, default=r'/home/ec2-user/zhou/VLM_KD/result/OpenGVLab/InternVL3_5-14B_timeblind.jsonl',
                        help='Path to the result file')
    parser.add_argument('--device_map', metavar='E', type=str,
                        default=r'cuda:1',
                        help='Device map')
    parser.add_argument('--judge_model', '-b', metavar='E', type=str, default="Qwen/Qwen3-14B", help='Judge model select.')
    parser.add_argument('--save_path', '-s', metavar='E', type=str, default='/home/ec2-user/zhou/VLM_KD/result/LLM_judge/LLM_judge_InternVL3_5-14B_timeblind.jsonl')
    return parser.parse_args()

def main():
    args = get_args()
    judge_model = Qwen3(model_name=args.judge_model, device_map=args.device_map)
    with open(args.result_file, "r", encoding="utf-8") as f:
        blind_data = [json.loads(line) for line in f if line.strip()]
    for i, item in enumerate(tqdm(blind_data, desc="Testing", total=len(blind_data))):
        if "predict_answer" in blind_data[i]:
            continue
        if "model_output" not in blind_data[i]:
            break
        prompt = create_extraction_prompt(model_output=blind_data[i]["model_output"], question_type=blind_data[i]["type"])
        response, thinking_content = judge_model.process_message(prompt=prompt)
        blind_data[i]['predict_answer'] = response
        if i % 200 == 0:
            with open(args.save_path, "w", encoding="utf-8") as f:
                for item in blind_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(args.save_path, "w", encoding="utf-8") as f:
        for item in blind_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()