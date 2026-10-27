# ============================================================
# GPT-2 AI Domain Text Generation - Flask Application
# ============================================================
# Run:
#     python app.py
# Then open:
#     http://127.0.0.1:5000
#
# Required model folder:
#     ./gpt2-ai-model
# ============================================================

import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# -----------------------------
# 1. Configuration
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gpt2-ai-model")

MAX_PROMPT_CHARACTERS = 2000
MAX_INPUT_TOKENS = 400
MAX_NEW_TOKENS = 180

# -----------------------------
# 2. Load the fine-tuned model
# -----------------------------

print("=" * 60)
print("Loading fine-tuned GPT-2...")
print("Model path:", MODEL_PATH)

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Model folder not found: {MODEL_PATH}\n"
        "Run train.py first and make sure it saves the model there."
    )

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# GPT-2 has no separate pad token by default.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Enable key/value caching for faster autoregressive generation.
model.config.use_cache = True

print("Model loaded successfully.")
print("Device:", device)
print("=" * 60)


# -----------------------------
# 3. Home page
# -----------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# -----------------------------
# 4. Generate text
# -----------------------------

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")

        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({
                "error": "Please enter a prompt."
            }), 400

        prompt = prompt.strip()

        if len(prompt) > MAX_PROMPT_CHARACTERS:
            return jsonify({
                "error": (
                    f"Please keep your prompt under "
                    f"{MAX_PROMPT_CHARACTERS} characters."
                )
            }), 400

        print("\n" + "=" * 60)
        print("PROMPT:", prompt)

        # IMPORTANT:
        # The training file contains ordinary AI articles, not
        # Question/Answer examples. Therefore, do not add a new
        # template such as "Question about Artificial Intelligence:
        # ... Answer:" at inference time. Pass the user's prompt
        # directly so it matches the training format more closely.

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            add_special_tokens=False,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        input_length = inputs["input_ids"].shape[1]
        if input_length == 0:
            return jsonify({"error": "The prompt could not be tokenized."}), 400

        # Sampling settings control variation and repetition.
        # They improve decoding behavior but cannot compensate
        # for a small or unsuitable training dataset.
        generation_config = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "temperature": 0.75,
            "top_k": 50,
            "top_p": 0.92,
            "repetition_penalty": 1.12,
            "no_repeat_ngram_size": 3,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **generation_config,
            )

        # Return only newly generated tokens, not the original prompt.
        new_tokens = output[0, input_length:]
        generated_text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if not generated_text:
            generated_text = (
                "The model did not generate additional text. "
                "Try a shorter or more specific prompt."
            )

        print("GENERATED TEXT:", generated_text)
        print("=" * 60)

        return jsonify({
            "generated_text": generated_text,
            "domain_rejected": False,
        }), 200

    except Exception as error:
        app.logger.exception("Text generation failed")
        return jsonify({
            "error": (
                "Text generation failed. Check the application "
                "terminal for the detailed error."
            )
        }), 500


# -----------------------------
# 5. Run Flask
# -----------------------------

if __name__ == "__main__":
    print("\nGPT-2 AI Text Generation App")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )
