import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# commented out because no cuda availability :(
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#   print("GPU:", torch.cuda.get_device_name(0))

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map="cpu", # can change to auto once cuda is found
    torch_dtype=torch.float16
)

text = "Hello, how are"
inputs = tokenizer(text, return_tensors="pt").to("cpu")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))