from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("Loading GPT-2...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

print("GPT-2 loaded successfully!")