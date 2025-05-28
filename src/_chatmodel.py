from typing import List

from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseChatModel:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def generate_single_turn_response(self, user_input: List[str], batch_size: int = 16):

        all_responses = []
        
        for start_idx in tqdm(range(0, len(user_input), batch_size), desc="Generating responses"):

            texts = user_input[start_idx: start_idx + batch_size]
            prompt_inputs = self.tokenizer(
                    text=texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                ).to(self.device)
            
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    
            prompt_completion_ids = self.model.generate(
                prompt_ids, attention_mask=prompt_mask, max_new_tokens=64
            )
            
            responses = self.tokenizer.batch_decode(prompt_completion_ids, skip_special_tokens=True)

            all_responses.extend(responses)

        return all_responses


class AdapterChatModel(BaseChatModel):
    def __init__(self, base_model_name="Qwen/Qwen3-1.7B", adapter_path="Chaew00n/test-supervised-fine-tuning",
                 device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = peft_model.merge_and_unload()
        self.model = self.model.to(device)


class PseudoChatModel(BaseChatModel):
    def __init__(self):
        pass

    def generate_single_turn_response(self, user_input: List[str], batch_size: int = 16):
        return [""] * len(user_input)


if __name__ == "__main__":
    chat_model = AdapterChatModel(
        base_model_name="Qwen/Qwen3-1.7B",
        adapter_path="Chaew00n/test-policy-optimization"
    )
    user_input = "Create similar query for 'How can I pass the exam?"
    response = chat_model.generate_single_turn_response(user_input)
    print(f"모델 응답: {response}")
