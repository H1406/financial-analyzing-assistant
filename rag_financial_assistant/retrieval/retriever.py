from retrieval.vector_store import VectorStore
from retrieval.embedding_model import EmbeddingModel
import yaml
import time
import mlflow
class Retriever:
    def __init__(self, config): 
        self.config = config
        self.model = EmbeddingModel(self.config["embedding_model"])
        self.vector_store = VectorStore.load("data/processed")
        self.top_k = self.config["top_k"]

    def preprocess_query(self, query):
        return query.strip().lower()
    def retrieve(self, query):
        start = time.time()
        query_emb = self.model.embed_query(query)
        results = self.vector_store.search(query_emb, self.top_k)
        latency = time.time() - start
        mlflow.log_metric("latency", latency)
        return results