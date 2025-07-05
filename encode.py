import os, json, pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import pandas as pd
model_path = r"C:\Users\GIGABYTE\Downloads\model_train"
model = SentenceTransformer('NghiemAbe/Vi-Legal-Bi-Encoder-v2')

def encode_documents(corpus_path, save_path):
    #with open(corpus_path,'rb') as f:
        #data = pickle.load(f)
    data=pd.read_csv(corpus_path)
    #corpus = list(data.values())
    corpus=list(data['text'])
    print(corpus[:5])
    corpus_embedding = model.encode(corpus, show_progress_bar=True, convert_to_tensor=True)
    
    idx_corpus = list(range(len(corpus)))
    
    with open(save_path, "wb") as f:
        pickle.dump((idx_corpus, corpus_embedding.cpu().numpy()), f)
    
    print(f"Encoded {len(corpus)} documents and saved to {save_path}")
if __name__ == "__main__":
    #corpus_path = r"C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\processed\corpus_after.pkl"
    #save_path = r"C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\processed\corpus_embedding"
    corpus_path=r"C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\data4pr\corpora.csv"
    save_path = r"C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\processed\corpus_embedding_base"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    encode_documents(corpus_path, save_path)