# ============================================================
# GPT-2 Fine-Tuning on an AI Domain Text Dataset
# ============================================================
# Run from your project folder:
#     python train.py
#
# Dataset:
#     C:/Finetune-AI-article/AI_domain_text_dataset_cleaned.txt
#
# Output model:
#     ./gpt2-ai-model
# ============================================================

import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)

# -----------------------------
# 1. Configuration
# -----------------------------

DATASET_PATH = r"C:/Finetune-AI-article/AI_domain_text_dataset_cleaned.txt"
MODEL_NAME = "gpt2"
OUTPUT_DIR = "./gpt2-ai-model"

BLOCK_SIZE = 256        # Try 256 first; use 128 if RAM is limited
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
VALIDATION_SIZE = 0.10
SEED = 42

set_seed(SEED)

# -----------------------------
# 2. Check dataset file
# -----------------------------

if not os.path.isfile(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset was not found at:\n{DATASET_PATH}\n"
        "Update DATASET_PATH near the top of train.py."
    )

print("=" * 65)
print("GPT-2 AI DOMAIN FINE-TUNING")
print("=" * 65)
print("Dataset:", DATASET_PATH)
print("Output:", OUTPUT_DIR)
print("Block size:", BLOCK_SIZE)
print("Device:", "CUDA GPU" if torch.cuda.is_available() else "CPU")

# -----------------------------
# 3. Load and clean the dataset
# -----------------------------
# The text file is loaded line by line. Empty lines are removed.
# We split the text rows BEFORE tokenization so that validation
# examples are not used to train the model.

raw_dataset = load_dataset(
    "text",
    data_files={"train": DATASET_PATH},
    split="train",
)

raw_dataset = raw_dataset.filter(
    lambda row: bool(row["text"] and row["text"].strip())
)

if len(raw_dataset) < 10:
    raise ValueError(
        "The dataset contains too few non-empty lines to train reliably."
    )

split_dataset = raw_dataset.train_test_split(
    test_size=VALIDATION_SIZE,
    seed=SEED,
)

train_text = split_dataset["train"]
validation_text = split_dataset["test"]

print("\nDataset loaded.")
print("Non-empty text lines:", len(raw_dataset))
print("Training lines:", len(train_text))
print("Validation lines:", len(validation_text))

# -----------------------------
# 4. Load tokenizer and model
# -----------------------------

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False  # Recommended during training

print("\nGPT-2 tokenizer and model loaded.")
print("Model parameters:", f"{model.num_parameters():,}")

# -----------------------------
# 5. Tokenize the text
# -----------------------------
# Each non-empty line receives an EOS token so the model can learn
# where one text line ends. We do NOT truncate each line to 128 tokens.
# Long lines are split into BLOCK_SIZE chunks in the next step.

def tokenize_function(batch):
    encoded = tokenizer(
        batch["text"],
        add_special_tokens=False,
        truncation=False,
    )

    all_input_ids = []
    all_attention_masks = []

    for ids in encoded["input_ids"]:
        ids = ids + [tokenizer.eos_token_id]
        all_input_ids.append(ids)
        all_attention_masks.append([1] * len(ids))

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
    }


tokenized_train = train_text.map(
    tokenize_function,
    batched=True,
    remove_columns=train_text.column_names,
    desc="Tokenizing training text",
)

tokenized_validation = validation_text.map(
    tokenize_function,
    batched=True,
    remove_columns=validation_text.column_names,
    desc="Tokenizing validation text",
)

# -----------------------------
# 6. Group tokens into fixed-size sequences
# -----------------------------
# GPT-2 learns next-token prediction from sequences. Grouping tokens
# gives it consistent training examples and avoids truncating every
# paragraph independently.

def group_texts(examples):
    concatenated = {}
    for key in examples.keys():
        values = []
        for sequence in examples[key]:
            values.extend(sequence)
        concatenated[key] = values

    total_length = len(concatenated["input_ids"])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    if total_length == 0:
        return {"input_ids": [], "attention_mask": []}

    result = {}
    for key, values in concatenated.items():
        result[key] = [
            values[i : i + BLOCK_SIZE]
            for i in range(0, total_length, BLOCK_SIZE)
        ]

    return result


lm_train_dataset = tokenized_train.map(
    group_texts,
    batched=True,
    desc=f"Grouping training tokens into {BLOCK_SIZE}-token sequences",
)

lm_validation_dataset = tokenized_validation.map(
    group_texts,
    batched=True,
    desc=f"Grouping validation tokens into {BLOCK_SIZE}-token sequences",
)

if len(lm_train_dataset) == 0 or len(lm_validation_dataset) == 0:
    raise ValueError(
        "Not enough tokens to create training/validation sequences. "
        "Reduce BLOCK_SIZE to 128 or add more text."
    )

print("\nTokenization and sequence grouping completed.")
print("Training sequences:", len(lm_train_dataset))
print("Validation sequences:", len(lm_validation_dataset))
print("Tokens per sequence:", BLOCK_SIZE)

# -----------------------------
# 7. Data collator
# -----------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2 uses causal language modeling, not masked LM
)

# -----------------------------
# 8. Training arguments
# -----------------------------
# Evaluation and saving happen at the end of each epoch.
# The best checkpoint is selected using validation loss.

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
    fp16=torch.cuda.is_available(),
)

# -----------------------------
# 9. Create Trainer
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_validation_dataset,
    data_collator=data_collator,
)

# -----------------------------
# 10. Train
# -----------------------------

print("\n" + "=" * 65)
print("Starting fine-tuning...")
print("=" * 65)

train_result = trainer.train()

print("\nTraining finished.")
print("Training metrics:", train_result.metrics)

# -----------------------------
# 11. Evaluate
# -----------------------------

print("\nEvaluating the selected model checkpoint...")
evaluation_metrics = trainer.evaluate()
print("Evaluation metrics:", evaluation_metrics)

eval_loss = evaluation_metrics.get("eval_loss")
if eval_loss is not None:
    try:
        perplexity = torch.exp(torch.tensor(eval_loss)).item()
        print("Validation perplexity:", round(perplexity, 2))
    except OverflowError:
        print("Validation perplexity was too large to calculate.")

# -----------------------------
# 12. Save final model and tokenizer
# -----------------------------
# trainer.save_model saves the best model when
# load_best_model_at_end=True is enabled.

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Restore cache setting for generation.
model.config.use_cache = True
model.config.pad_token_id = tokenizer.pad_token_id
model.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 65)
print("FINE-TUNING COMPLETE")
print("=" * 65)
print("Saved model and tokenizer to:", os.path.abspath(OUTPUT_DIR))
print("\nUse this same folder as MODEL_PATH in app.py.")
