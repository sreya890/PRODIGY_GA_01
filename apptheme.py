
from flask import Flask, render_template, request, jsonify

from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch


# ============================================================
# Flask Application
# ============================================================

app = Flask(__name__)


# ============================================================
# Load Fine-Tuned GPT-2
# ============================================================

MODEL_PATH = "./gpt2-ai-model"

print("========================================")
print("Loading Fine-Tuned GPT-2...")
print("========================================")

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


print("GPT-2 model loaded successfully!")
print("Device:", device)


# ============================================================
# Home Page
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# Generate Text
# ============================================================

@app.route("/generate", methods=["POST"])
def generate():

    try:

        # ----------------------------------------------------
        # Get prompt from frontend
        # ----------------------------------------------------

        data = request.get_json()

        prompt = data.get("prompt", "").strip()


        # ----------------------------------------------------
        # Validate prompt
        # ----------------------------------------------------

        if not prompt:

            return jsonify({
                "error": "Please enter a prompt."
            }), 400


        print("\n========================================")
        print("PROMPT RECEIVED")
        print("========================================")

        print(prompt)


        # ----------------------------------------------------
        # Tokenize prompt
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

                # Maximum number of NEW tokens
                max_new_tokens=150,

                # Generate one response
                num_return_sequences=1,

                # Controls randomness
                temperature=0.7,

                # Consider top 40 tokens
                top_k=40,

                # Nucleus sampling
                top_p=0.90,

                # Reduce repeated text
                repetition_penalty=1.15,

                # Prevent repeated 3-word combinations
                no_repeat_ngram_size=3,

                # Enable sampling
                do_sample=True,

                # Padding
                pad_token_id=tokenizer.pad_token_id
            )


        # ----------------------------------------------------
        # Decode generated tokens
        # ----------------------------------------------------

        generated_text = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )


        # ----------------------------------------------------
        # Remove prompt from response
        # ----------------------------------------------------

        if generated_text.startswith(prompt):

            generated_text = generated_text[
                len(prompt):
            ].strip()


        # ----------------------------------------------------
        # Keep complete sentences
        # ----------------------------------------------------

        last_full_stop = generated_text.rfind(".")


        if last_full_stop != -1:

            generated_text = generated_text[
                :last_full_stop + 1
            ]


        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if not generated_text:

            generated_text = (
                "The model was unable to generate "
                "a complete response."
            )


        print("\n========================================")
        print("GENERATED RESPONSE")
        print("========================================")

        print(generated_text)


        # ----------------------------------------------------
        # Send response to frontend
        # ----------------------------------------------------

        return jsonify({

            "generated_text": generated_text

        })


    except Exception as e:

        print("\nERROR:")
        print(str(e))


        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("GPT-2 AI APPLICATION")
    print("========================================")

    print("Open: http://127.0.0.1:5000")


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )