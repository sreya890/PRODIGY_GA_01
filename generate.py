# ============================================================
# Generate text using the fine-tuned GPT-2 model
# ============================================================

from transformers import GPT2Tokenizer, GPT2LMHeadModel


# -----------------------------------
# 1. Load fine-tuned model
# -----------------------------------

model_path = "./gpt2-ai-model"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)

model = GPT2LMHeadModel.from_pretrained(model_path)

print("Fine-tuned GPT-2 model loaded successfully!")


# -----------------------------------
# 2. Set padding token
# -----------------------------------

tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id


# -----------------------------------
# 3. Give a prompt
# -----------------------------------

prompt = "A production AI system receives patient data, retrieves relevant medical information from a vector database, uses a machine learning model to make a prediction, and presents the result through a web application. The complete workflow should"

# -----------------------------------
# 4. Convert prompt into tokens
# -----------------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


# -----------------------------------
# 5. Generate text
# -----------------------------------

output = model.generate(
    **inputs,
    max_new_tokens=150,
    num_return_sequences=1,
    temperature=0.7,
    top_k=40,
    top_p=0.90,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)

generated_text = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

# Find the last complete sentence
last_full_stop = generated_text.rfind(".")

if last_full_stop != -1:
    generated_text = generated_text[:last_full_stop + 1]

print("\n========================================")
print("PROMPT")
print("========================================")
print(prompt)

print("\n========================================")
print("GENERATED TEXT")
print("========================================")
print(generated_text)


