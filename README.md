# **📜 AI Legal Assistant – Misleading Clause Detector**

An AI-powered web application that analyzes Terms & Conditions or legal documents (PDF/TXT) and detects misleading clauses. The system uses SentenceTransformer embeddings, FAISS similarity search, and an optional fine-tuned BERT classifier to identify risky or biased clauses.

## 🚀 Features

- 📂 Upload PDF or TXT contracts.

- 🔎 Detects misleading or biased clauses automatically.

- ✨ Highlights detected clauses directly inside the document.

- 📌 Provides a summary list of misleading clauses.

- ⚡ Uses FAISS + SentenceTransformer for fast similarity search.

- 🤖 Supports fine-tuned BERT classifier for better accuracy.

- 🌐 Web app built with Flask + HTML/CSS frontend.

## 🛠️ Tech Stack

- Backend: Python, Flask

- NLP Models: Hugging Face Transformers, Sentence-Transformers

- Vector Search: FAISS

- Frontend: HTML5, CSS3 (custom UI with gradient + animations)

- Libraries: PyPDF2, NLTK, NumPy, Pandas

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/swalhasakeer/AI-Legal-Assistant-for-Detecting-Misleading-Clauses-in-Contracts.git
cd AI-Legal-Assistant-for-Detecting-Misleading-Clauses-in-Contracts
```


### 2️⃣ Create a virtual environment 

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```


### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

1. Run Flask app:

```python
python app.py
```

2. Open in browser:

```bash
http://127.0.0.1:5000/
```

3. Upload a .pdf or .txt file.

4. View results:

   - 📄 Document text with highlighted misleading clauses.

   - 📌 Summary list of all detected clauses.


## Demo

![Project demo1](https://github.com/user-attachments/assets/cfed380c-5567-43f3-b470-cbbff6ad0cc2)


  
## 🔮 Future Improvements

✅ Mobile/Cloud deployment.

✅ Advanced explanation.

✅ Multi-language support.

✅ Enterprise integration.

✅ Clause recommentation engine
