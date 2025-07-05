import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from main.retrieval.faiss.faiss_search import FaissFlatSearcher
from main.question_answering.qa_bot import QABot
import traceback

# --- Khởi tạo Flask ---
app = Flask(__name__)
CORS(app)

# --- Đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "model")
QA_MODEL_PATH = os.path.join(BASE_DIR, "model_qa")
CORPUS_TEXT = os.path.join(BASE_DIR, "data", "processed", "corpus_after.pkl")
CORPUS_EMBEDDING = os.path.join(BASE_DIR, "data", "processed", "corpus_embedding")

# --- Load Sentence Embedding model ---
print("🔹 Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

# --- Load corpus và FAISS index ---
print("🔹 Loading corpus and embeddings...")
#with open(r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\ChatLaw\backend\data\data4pr\corpora.csv', "rb") as f:
    #corpus_data = pickle.load(f)
corpus_data=pd.read_csv(r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\ChatLaw\backend\data\data4pr\corpora.csv')
print(corpus_data)
corpus_list = list(corpus_data['text'])
print(len(corpus_list))
with open(CORPUS_EMBEDDING, "rb") as f:
    idx_corpus, corpus_embedding = pickle.load(f)
corpus_embedding = corpus_embedding.astype("float32")

searcher = FaissFlatSearcher(dim=corpus_embedding.shape[1])
searcher.add(corpus_embedding)

# --- Load QA Model một lần ---
print("🔹 Loading QA model...")
bot = QABot('kienhoang123/vietnamese-legal-gpt2')
bot.load_model()

# --- ROUTES ---
@app.route("/", methods=["GET"])
def index():
    return "🧾 Chatbot Thuế Việt Nam đã sẵn sàng!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Bạn cần gửi một câu hỏi qua trường 'question'."}), 400

    question = data["question"].strip()

    if question == "":
        return jsonify({"error": "Câu hỏi không được để trống."}), 400

    try:
        # --- Bước 1: Tìm context ---
        corpus_relavant=[]
        processed_q = searcher.preprocess(question)
        query_embedding = embedding_model.encode(processed_q).astype("float32").reshape(1, -1)
        scores, indices = searcher.search(query_embedding, 5)
        for rank, (doc_idx, score) in enumerate(zip(indices[0], scores), 1):
                print(doc_idx)
                doc = corpus_list[doc_idx]
                corpus_relavant.append(doc)
        indices = indices.flatten()
        context = "\n".join([corpus_list[i] for i in indices]).strip()

        if not context:
            return jsonify({
                "question": question,
                "answer": "Xin lỗi, tôi chưa có dữ liệu phù hợp để trả lời câu hỏi này.",
                "context": ""
            })

        # --- Bước 2: Gọi model sinh câu trả lời ---
        answer = bot.generate_answer(corpus_relavant,question)
        corpus_relavant=[]
        return jsonify({
            "question": question,
            "context": context,
            "answer": answer
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- CHẠY ---
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False) 
