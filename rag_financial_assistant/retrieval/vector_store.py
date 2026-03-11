import faiss
import numpy as np
import pickle


class VectorStore:

    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, metadata):

        self.index.add(np.array(embeddings))
        self.metadata.extend(metadata)

    def save(self, path):

        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)