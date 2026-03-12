from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=True)
    
    def embed_query(self, query):
        return self.model.encode(query, show_progress_bar=True)  

# model = EmbeddingModel("all-MiniLM-L6-v2")
# emb = model.embed(["test sentence"])
# print(emb.shape)