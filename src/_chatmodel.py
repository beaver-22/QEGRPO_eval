from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseChatModel:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def generate_single_turn_response(self, user_input):
        messages = [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        response_ids = self.model.generate(**inputs, max_new_tokens=256)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return " " + response


class AdapterChatModel(BaseChatModel):
    def __init__(self, base_model_name="Qwen/Qwen3-1.7B", adapter_path="Chaew00n/test-supervised-fine-tuning",
                 device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = peft_model.merge_and_unload()
        self.model = self.model.to(device)

    def generate_single_turn_response(self, user_input):
        messages = [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        response_ids = self.model.generate(**inputs, max_new_tokens=1024)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return " " + response


class PseudoChatModel(BaseChatModel):
    def __init__(self):
        pass

    def generate_single_turn_response(self, user_input):
        return ""


if __name__ == "__main__":
    chat_model = AdapterChatModel(
        base_model_name="Qwen/Qwen3-1.7B",
        adapter_path="Chaew00n/test-policy-optimization"
    )
    user_input = "Create similar query for 'How can I pass the exam?"
    response = chat_model.generate_single_turn_response(user_input)
    print(f"모델 응답: {response}")
