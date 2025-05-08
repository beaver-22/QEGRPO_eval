from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel

# Base + Adapter 불러오기
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").to("cuda:1")
peft_model = PeftModel.from_pretrained(base_model, "Chaew00n/test-supervised-fine-tuning")
peft_model2 = PeftModel.from_pretrained(base_model, "Chaew00n/test-supervised-fine-tuning")

# Merge
merged_model = peft_model.merge_and_unload()

# Merge된 모델을 로컬에 저장
local_dir = "./merged-qwen1"
merged_model.save_pretrained(local_dir)

# Tokenizer도 저장 (중요)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer.save_pretrained(local_dir)


