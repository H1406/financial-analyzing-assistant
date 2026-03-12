import faiss
import numpy as np
import pickle


class VectorStore:

    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata


    @classmethod
    def create(cls, dim):
        index = faiss.IndexFlatL2(dim)
        metadata = []
        return cls(index, metadata)


    @classmethod
    def load(cls, path):

        index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return cls(index, metadata)


    def add(self, embeddings, metadata):

        embeddings = np.array(embeddings)

        self.index.add(embeddings)

        self.metadata.extend(metadata)


    def search(self, query_embedding, top_k=5):

        query_embedding = np.array([query_embedding])

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for i, idx in enumerate(indices[0]):

            doc = self.metadata[idx]

            results.append({
                "text": doc["text"],
                "source": doc["source"],
                "page": doc["page"],
                "score": float(distances[0][i])
            })

        return results


    def save(self, path):

        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)