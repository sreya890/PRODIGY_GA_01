from flask import Flask, render_template, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
import torch

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# ============================================================
# LOAD FINE-TUNED GPT-2
# ============================================================

MODEL_PATH = "./gpt2-ai-model"

print("=" * 50)
print("Loading Fine-Tuned GPT-2...")
print("=" * 50)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Configure padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Use CPU
device = torch.device("cpu")
model.to(device)
model.eval()

print("GPT-2 model loaded successfully!")
print("Device:", device)

# ============================================================
# SEMANTIC AI-DOMAIN CLASSIFIER
# ============================================================

print("=" * 50)
print("Loading semantic AI-domain classifier...")
print("=" * 50)

classifier = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Representative AI-domain examples
AI_DOMAIN_EXAMPLES = [
    "Explain how computers learn patterns from data.",
    "How can a machine make predictions using past examples?",
    "Explain neural networks and how they learn.",
    "How does a chatbot generate human-like text?",
    "How can AI understand and classify images?",
    "Explain how language models process human language.",
    "How do we train, evaluate, and deploy a predictive model?",
    "How can data science help discover patterns in datasets?",
    "How does a retrieval augmented generation system work?",
    "How can a computer recognize speech?",
    "How can machine learning identify unusual patterns in data?",
    "Explain how AI can be integrated into a web application.",
    "How can artificial intelligence help doctors diagnose diseases?",
    "Explain supervised and unsupervised learning.",
    "How can a computer generate text from a prompt?",
    "How do we prepare data for machine learning?",
    "Explain natural language processing with examples.",
    "How can a model predict numerical values?",
    "What is deep learning and how does it work?",
    "How can AI systems retrieve information from documents?"
]

# Encode examples only once at startup
AI_DOMAIN_EMBEDDINGS = classifier.encode(
    AI_DOMAIN_EXAMPLES,
    convert_to_tensor=True,
    normalize_embeddings=True
)

# Initial threshold; calibrate using your own test prompts
AI_DOMAIN_THRESHOLD = 0.40

DEFAULT_MESSAGE = (
    "I'm designed to answer questions related to Artificial "
    "Intelligence, Machine Learning, Deep Learning, Generative "
    "AI, and Data Science. Please try asking a question within "
    "these areas."
)


def is_ai_domain(prompt):
    """
    Compare the meaning of a prompt with representative
    AI-domain examples using semantic similarity.
    """

    prompt_embedding = classifier.encode(
        prompt,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(
        prompt_embedding,
        AI_DOMAIN_EMBEDDINGS
    )[0]

    best_score = similarities.max().item()

    print("AI-domain similarity score:", round(best_score, 3))

    return best_score >= AI_DOMAIN_THRESHOLD


print("Semantic classifier loaded successfully!")

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# GENERATE TEXT
# ============================================================

@app.route("/generate", methods=["POST"])
def generate():

    try:
        # Get prompt from frontend
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")

        # Validate prompt type and content
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({
                "error": "Please enter a valid prompt."
            }), 400

        prompt = prompt.strip()

        # Prevent excessively long prompts
        if len(prompt) > 2000:
            return jsonify({
                "error": "Please keep your prompt under 2000 characters."
            }), 400

        print("\n" + "=" * 50)
        print("PROMPT RECEIVED")
        print("=" * 50)
        print(prompt)

        # ----------------------------------------------------
        # SEMANTIC DOMAIN CHECK
        # ----------------------------------------------------

        if not is_ai_domain(prompt):
            print("Prompt rejected: outside the supported domain.")

            return jsonify({
                "generated_text": DEFAULT_MESSAGE,
                "domain_rejected": True
            }), 200

        print("Prompt accepted: AI-domain related.")

        # ----------------------------------------------------
                # ----------------------------------------------------
        # PREPARE AND TOKENIZE PROMPT
        # ----------------------------------------------------

        model_prompt = (
            "Question about Artificial Intelligence: "
            + prompt.strip()
            + "\nAnswer: "
        )

        inputs = tokenizer(
            model_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # GENERATE TEXT USING FINE-TUNED GPT-2
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

        # Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_length:]

        generated_text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        # Keep the original complete-sentence behavior
        last_full_stop = generated_text.rfind(".")

        if last_full_stop != -1:
            generated_text = generated_text[:last_full_stop + 1]

        if not generated_text:
            generated_text = (
                "The model was unable to generate a complete "
                "response. Please try rephrasing your question."
            )

        print("\n" + "=" * 50)
        print("GENERATED RESPONSE")
        print("=" * 50)
        print(generated_text)

        return jsonify({
            "generated_text": generated_text,
            "domain_rejected": False
        }), 200

    except Exception as e:
        print("\nERROR:", str(e))

        return jsonify({
            "error": "An error occurred while processing your prompt."
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("GPT-2 AI APPLICATION")
    print("=" * 50)
    print("Open: http://127.0.0.1:5000")

    # Disable the reloader to avoid loading both models twice
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
