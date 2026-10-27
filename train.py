# ============================================================
# GPT-2 Fine-Tuning on AI Domain Text Dataset
# ============================================================


# -----------------------------------
# 1. Import libraries
# -----------------------------------

from datasets import load_dataset

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)


# -----------------------------------
# 2. Load dataset
# -----------------------------------

dataset = load_dataset(
    "text",
    data_files="C:/Finetune-AI-article/AI_domain_text_dataset_cleaned.txt"
)

# Remove empty lines
dataset["train"] = dataset["train"].filter(
    lambda example: example["text"].strip() != ""
)

print("Dataset loaded successfully!")
print(dataset)


# -----------------------------------
# 3. Split dataset
# -----------------------------------

dataset = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = dataset["train"]
validation_dataset = dataset["test"]

print("\nTraining samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))


# -----------------------------------
# 4. Load GPT-2 tokenizer
# -----------------------------------

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 does not have a padding token by default
# We use EOS token as padding token
tokenizer.pad_token = tokenizer.eos_token

print("\nGPT-2 tokenizer loaded successfully!")


# -----------------------------------
# 5. Load GPT-2 model
# -----------------------------------

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set padding token
model.config.pad_token_id = tokenizer.pad_token_id

print("\nGPT-2 model loaded successfully!")


# -----------------------------------
# 6. Tokenize dataset
# -----------------------------------

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

print("\nTokenization completed!")


# -----------------------------------
# 7. Create data collator
# -----------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

print("\nData collator created successfully!")


# -----------------------------------
# 8. Display dataset/model information
# -----------------------------------

print("\nExample original text:")
print(train_dataset[0]["text"])

print("\nExample token IDs:")
print(tokenized_dataset["train"][0]["input_ids"][:20])

print("\nNumber of model parameters:")
print(model.num_parameters())

print("\nModel type:")
print(type(model))


# -----------------------------------
# 9. Training configuration
# -----------------------------------

training_args = TrainingArguments(

    # Where the trained model/checkpoints will be stored
    output_dir="./gpt2-ai-model",

    # Number of times the model sees the entire dataset
    num_train_epochs=3,

    # Small batch size for limited RAM
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    # Accumulate gradients for 4 steps
    # Effective batch size = 1 × 4 = 4
    gradient_accumulation_steps=4,

    # Evaluate after every epoch
    eval_strategy="epoch",

    # Save model after every epoch
    save_strategy="epoch",

    # Keep only the latest 2 checkpoints
    save_total_limit=2,

    # Display training information every 10 steps
    logging_steps=10,

    # Learning rate
    learning_rate=5e-5,

    # Don't send logs to external services
    report_to="none",

    # Load best checkpoint after training
    load_best_model_at_end=True
)

print("\nTraining arguments created successfully!")


# -----------------------------------
# 10. Create Trainer
# -----------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator
)
print("\nTrainer created successfully!")


# -----------------------------------
# 11. Start training
# -----------------------------------

print("\n========================================")
print("Starting GPT-2 fine-tuning...")
print("========================================\n")

trainer.train()

print("\n========================================")
print("Training completed!")
print("========================================")


# -----------------------------------
# 12. Save fine-tuned model
# -----------------------------------

model.save_pretrained("./gpt2-ai-model")

tokenizer.save_pretrained("./gpt2-ai-model")

print("\nFine-tuned model saved successfully!")

print("\nModel location:")
print("./gpt2-ai-model")


# -----------------------------------
# 13. Final message
# -----------------------------------

print("\n========================================")
print("GPT-2 FINE-TUNING FINISHED")
print("========================================")