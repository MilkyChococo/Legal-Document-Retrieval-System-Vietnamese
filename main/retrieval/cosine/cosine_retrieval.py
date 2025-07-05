import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class EmbeddingRetrieval():
    def __init__(self, query_embedding, corpus_embeddings, k):
        self.query_embedding = query_embedding
        self.corpus_embeddings = corpus_embeddings
        self.k = k

    def retrieval_module(self):
        similarities = cosine_similarity(self.query_embedding, self.corpus_embeddings)
        
        similarities = similarities.flatten()
        top_k_indices = similarities.argsort()[::-1][:self.k]
        
        return top_k_indices, similarities[top_k_indices]