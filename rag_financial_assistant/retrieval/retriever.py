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
        retrieval_config = self.config.get("retrieval", {})
        self.vector_weight = retrieval_config.get("vector_weight", self.config.get("vector_weight", 0.6))
        self.bm25_weight = retrieval_config.get("bm25_weight", self.config.get("bm25_weight", 0.4))

    def preprocess_query(self, query):
        return query.strip().lower()
    def retrieve(self, query):
        start = time.time()
        query_emb = self.model.embed_query(query)
        results = self.vector_store.search(
            query_embedding=query_emb,
            query_text=query,
            top_k=self.top_k,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
        )
        latency = time.time() - start
        mlflow.log_metric("latency", latency)
        return results
