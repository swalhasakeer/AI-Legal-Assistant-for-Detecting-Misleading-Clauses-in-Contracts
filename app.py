import os
import re
import faiss
import nltk
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader

# --------------------- Setup ---------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# NLTK sentence tokenizer
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

# Embeddings for FAISS
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Classifier (optional, your fine-tuned model)
MODEL_PATH = r"C:\Users\Pc\OneDrive\Desktop\AI Legal Assistant\gen_ai_model"
USE_CLASSIFIER = True
try:
    classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)
except Exception as e:
    print("[DEBUG] Could not load fine-tuned classifier, continuing with FAISS only:", e)
    USE_CLASSIFIER = False
    classifier = None

# --------------------- Common misleading clauses ---------------------
COMMON_MISLEADING = [
    "The company may terminate the agreement at any time without prior notice.",
    "We reserve the right to modify terms without informing you.",
    "Refunds will not be provided under any circumstances.",
    "Your personal data may be shared with third parties without consent.",
    "Access to the service may be suspended or withdrawn at any time without explanation.",
    "A refund will only be issued under conditions determined solely by the company.",
    "We may change prices or fees at any time without prior notification.",
    "The company is not responsible for any loss or damage, whether direct or indirect.",
]

seed_embeddings = embed_model.encode(COMMON_MISLEADING, convert_to_numpy=True)
dimension = seed_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(seed_embeddings)

# --------------------- Helpers ---------------------
def extract_text_from_pdf(filepath: str) -> str:
    text = ""
    reader = PdfReader(filepath)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def safe_sent_tokenize(text: str):
    try:
        sents = sent_tokenize(text)
        if sents:
            return sents
    except Exception as e:
        print("[DEBUG] sent_tokenize failed, fallback regex:", e)
    return re.split(r'(?<=[\.\!\?])\s+', text)

def find_misleading(sent: str, threshold: float = 0.85):
    emb = embed_model.encode([sent], convert_to_numpy=True)
    distances, idxs = faiss_index.search(emb, 1)
    best_distance = float(distances[0][0])
    best_idx = int(idxs[0][0])
    if best_distance < threshold:
        return COMMON_MISLEADING[best_idx]
    return None

def highlight_clauses(full_text: str, clauses: list[str]) -> str:
    highlighted = full_text
    for clause in sorted(set(clauses), key=len, reverse=True):
        if not clause.strip():
            continue
        pattern = re.compile(re.escape(clause), re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f"<mark style='background-color:#ffeb3b; padding:2px 3px; border-radius:3px; font-weight:600;'>{m.group(0)}</mark>",
            highlighted
        )
    return highlighted

# --------------------- Routes ---------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if not file.filename:
            return render_template("index.html", error="No file selected.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # extract text
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith(".txt"):
            text = extract_text_from_txt(filepath)
        else:
            return render_template("index.html", error="Unsupported file type. Use .pdf or .txt")

        if not text.strip():
            return render_template("result.html", result="❌ Could not extract readable text.", highlighted_text="", detected=[], matched=[])

        sentences = [s.strip() for s in safe_sent_tokenize(text) if s.strip()]
        detected, matched = [], []

        for sent in sentences:
            matched_clause = find_misleading(sent)
            if matched_clause:
                detected.append(sent)
                matched.append(matched_clause)

        if detected:
            highlighted_text = highlight_clauses(text, detected)
            return render_template(
                "result.html",
                result="📄 Detected misleading clauses:",
                highlighted_text=highlighted_text,
                detected=detected,
                matched=matched,
                detected_joined="|||".join(detected),
                matched_joined="|||".join(matched)
            )
        else:
            return render_template(
                "result.html",
                result="✅ No misleading clauses found.",
                highlighted_text=text,
                detected=[],
                matched=[],
                detected_joined="",
                matched_joined=""
            )

    return render_template("index.html")

@app.route("/summary", methods=["POST"])
def summary():
    matched_joined = request.form.get("matched", "")
    matched = [m for m in matched_joined.split("|||") if m.strip()]

    # ✅ Remove duplicates but keep order
    unique_matched = list(dict.fromkeys(matched))

    if not unique_matched:
        unique_matched = ["No misleading clauses found."]

    return render_template("summary.html", matched=unique_matched)


# --------------------- Run ---------------------
if __name__ == "__main__":
    app.run(debug=True)
