import sys
from main.retrieval.cosine.cosine_retrieval import EmbeddingRetrieval
import os, json, pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer,models
from main.preprocessor.preprocess import TextProcessor
import pandas as pd
corpus_select=r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\data4pr\corpora.csv'
test_select=r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\data4pr\test.csv'
model_path=r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\model'
ranking_path = 'main/evaluation/ranking_10_base_model.csv'
#model = SentenceTransformer(model_path)
model=SentenceTransformer("NghiemAbe/Vi-Legal-Bi-Encoder-v2")
####Preprocess Query and Evaluation Data
corpus=pd.read_csv(corpus_select)
test=pd.read_csv(test_select)
rank=pd.read_csv(ranking_path)
cid=list(corpus['cid'])
with open(r"data\processed\corpus_embedding_base", "rb") as f:
    idx_corpus, corpus_embedding = pickle.load(f)
corpus_embedding = corpus_embedding.astype("float32")
data_csv = []
preprocess=TextProcessor()
k = 10
for qid, query in tqdm(zip(test['qid'], test['question']), total=len(test)):
    top_cids = rank[rank['qid'] == qid].iloc[:k]['cid'].tolist()
    order= [cid.index(cid_value) for cid_value in top_cids]
    corpus_embeddings = corpus_embedding[order]
    query_user = query
    #query_user=preprocess.preprocess_text(query_user1)
    #query_user = preprocess.post_preprocess_text(query_user)
    query_embedding = model.encode(query_user).astype("float32")  
    query_embedding= query_embedding.reshape(1, -1)
    if query_embedding.shape[-1] != 768:
        query_embedding = query_embedding[:, -768:]

    searcher = EmbeddingRetrieval(corpus_embeddings, query_embedding,k)
    indices, scores = searcher.retrieval_module()
    indices = indices.flatten()
    scores  = scores.flatten()
    reranked_cids = [top_cids[i] for i in indices]
    for order, (doc_cid, score) in enumerate(zip(reranked_cids, scores), 1):
            score=score/100
            data_csv.append([qid,query_user, doc_cid, score])
df = pd.DataFrame(data_csv, columns=['qid','Question','cid', 'Score'])
df.to_csv('main/evaluation/ranking_after_10_base_model.csv', index=False)
    
