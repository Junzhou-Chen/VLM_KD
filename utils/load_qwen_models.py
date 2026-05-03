import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLMoeForConditionalGeneration


class Qwen3_VL:
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-VL-8B-Thinking",
                 att: str = "flash_attention_2",
                 dtype: str = "auto"):
        is_moe = re.search(r"A\d+B", model_name) is not None
        if is_moe:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                attn_implementation=att,
                device_map="auto",
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                attn_implementation=att,
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def process_message(self, message, max_new_tokens: int = 256):
        inputs = self.processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
