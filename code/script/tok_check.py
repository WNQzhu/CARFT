from transformers import AutoTokenizer

# load
tokenizer = AutoTokenizer.from_pretrained("/home/wnq/models/codellama/CodeLlama-7b-hf")

# eos
eos_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

#pad
pad_token = tokenizer.pad_token
pad_token_id = tokenizer.pad_token_id


print(f"(eos_token): {eos_token}")
print(f" ID (eos_token_id): {eos_token_id}")

print(f"(pad_token): {pad_token}")
print(f" ID (pad_token_id): {pad_token_id}")


