from main.retrieval.faiss.faiss_search import FaissFlatSearcher
import os, json, pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
#from main.question_answering.qa_bot import QABot
model_path = r"C:\Users\GIGABYTE\Downloads\model_train"

embedding_model = SentenceTransformer(model_path)

query_user1 = "Giáo viên nghỉ hè có được hưởng lương và hưởng phụ cấp theo quy định hiện hành không?"
with open(r"data\processed\corpus_embedding", "rb") as f:
    idx_corpus, corpus_embedding = pickle.load(f)
corpus_embedding = corpus_embedding.astype("float32")
with open(r"data\processed\corpus_after.pkl", "rb") as f:
    data = pickle.load(f)

searcher = FaissFlatSearcher(dim=corpus_embedding.shape[1])
searcher.add(corpus_embedding)
query_user=searcher.preprocess(query_user1)
query_embedding = embedding_model.encode(query_user).astype("float32")     
print(query_embedding.shape)   
query_embedding= query_embedding.reshape(1, -1)
k = 5
scores, indices = searcher.search(query_embedding, k)
indices = indices.flatten()
print(indices)
scores  = scores.flatten()
print("Câu hỏi:", query_user)
print("Kết quả trả lời:")
corpus_relavant=[]
data=list(data.values())
for rank, (doc_idx, score) in enumerate(zip(indices, scores), 1):
    doc = data[doc_idx]
    score=score/100
    corpus_relavant.append(doc)
    print(f"{rank}. {doc}  (Similarities: {score:.4f})")
#modelqa_path=r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval\model_qa'
#model_qa=QABot(modelqa_path,query_user1)
#model_qa.load_model()
#print(model_qa.generate_answer(corpus_relavant))
