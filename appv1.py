
from flask import Flask, render_template, request, jsonify

from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch


# ============================================================
# 1. Create Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. Load fine-tuned GPT-2 model
# ============================================================

MODEL_PATH = "./gpt2-ai-model"

print("Loading fine-tuned GPT-2 model...")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# GPT-2 does not have a padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Use CPU
device = torch.device("cpu")

model.to(device)

# Evaluation mode
model.eval()

print("Fine-tuned GPT-2 model loaded successfully!")


# ============================================================
# 3. Home page
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# 4. Generate text
# ============================================================

@app.route("/generate", methods=["POST"])
def generate():

    try:

        # Get JSON data from frontend
        data = request.get_json()

        prompt = data.get("prompt", "").strip()

        # Check whether prompt is empty
        if not prompt:
            return jsonify({
                "error": "Please enter a prompt."
            }), 400


        # ----------------------------------------------------
        # Convert prompt into tokens
        # ----------------------------------------------------

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # Generate text
        # ----------------------------------------------------

        with torch.no_grad():

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


        # ----------------------------------------------------
        # Convert tokens back to text
        # ----------------------------------------------------

        generated_text = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )


        # ----------------------------------------------------
        # Keep the last complete sentence
        # ----------------------------------------------------

        last_full_stop = generated_text.rfind(".")

        if last_full_stop != -1:

            generated_text = generated_text[
                :last_full_stop + 1
            ]


        # ----------------------------------------------------
        # Send result to frontend
        # ----------------------------------------------------

        return jsonify({
            "generated_text": generated_text
        })


    except Exception as e:

        print("Error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 5. Run Flask application
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
