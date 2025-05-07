from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseChatModel:
    def __init__(self, model_name="Qwen/Qwen3-1.7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_single_turn_response(self, user_input):
        messages = [{"role": "user", "content": user_input}]

        # Todo: We may need to add or change argument value "https://github.com/huggingface/transformers/blob/798f948e88fd0b93fc515ec6b96e0503b78ad6ba/src/transformers/tokenization_utils_base.py#L1530"
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Todo: We may need to add generation config "https://github.com/huggingface/transformers/blob/798f948e88fd0b93fc515ec6b96e0503b78ad6ba/src/transformers/generation/utils.py#L2146"
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response