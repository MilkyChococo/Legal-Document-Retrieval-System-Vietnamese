import sys
from main.retrieval.faiss.faiss_search import FaissFlatSearcher
import os, json, pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer,models
import pandas as pd
corpus_select=r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\data4pr\corpora.csv'
test_select=r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\data4pr\test.csv'
model_path=r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\model'

#model = SentenceTransformer(model_path)
model=SentenceTransformer("NghiemAbe/Vi-Legal-Bi-Encoder-v2")
####Preprocess Query and Evaluation Data
corpus=pd.read_csv(corpus_select)
test=pd.read_csv(test_select)
with open(r"data\processed\corpus_embedding_base", "rb") as f:
    idx_corpus, corpus_embedding = pickle.load(f)
corpus_embedding = corpus_embedding.astype("float32")
searcher = FaissFlatSearcher(dim=corpus_embedding.shape[1])
searcher.add(corpus_embedding)
data_csv = []
for qid, query in tqdm(zip(test['qid'], test['question']), total=len(test)):
    query_user = query
    #query_user=searcher.preprocess(query_user1)
    query_embedding = model.encode(query_user).astype("float32")  
    query_embedding= query_embedding.reshape(1, -1)
    if query_embedding.shape[-1] != 768:
        query_embedding = query_embedding[:, -768:]
    k = 10
    scores, indices = searcher.search(query_embedding, k)
    indices = indices.flatten()
    scores  = scores.flatten()
    corpus_relavant=[]
    data=list(corpus['text'])
    cid=list(corpus['cid'])
    for rank, (doc_idx, score) in enumerate(zip(indices, scores), 1):
            doc = data[doc_idx]
            doc_cid = cid[doc_idx]
            score=score/100
            corpus_relavant.append(doc)
            data_csv.append([qid,query_user, doc_cid, score])
df = pd.DataFrame(data_csv, columns=['qid','Question','cid', 'Score'])
df.to_csv('main/evaluation/ranking_10_base_model.csv', index=False)
    
