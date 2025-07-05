import faiss
import sys
import numpy as np
sys.path.append(r'C:\Users\GIGABYTE\OneDrive\Desktop\CS221_P21\Legal Retriever Model\vietnamese-legal-retrieval-main')
from main.preprocessor.preprocess import TextProcessor

class FaissFlatSearcher:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlat(dim)
    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)
    def preprocess(self,text):
        preprocess=TextProcessor()
        text=preprocess.preprocess_text(text)
        return preprocess.post_preprocess_text(text)
    def search(self, q_reps, k):
        return self.index.search(q_reps, k)