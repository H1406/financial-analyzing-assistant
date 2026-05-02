import pickle

import faiss
import numpy as np

from retrieval.bm25 import BM25Index


class VectorStore:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata
        self.bm25 = BM25Index()
        self._rebuild_bm25()

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

    def _rebuild_bm25(self):
        self.bm25 = BM25Index()
        texts = [doc.get("text", "") for doc in self.metadata]
        if texts:
            self.bm25.add_documents(texts)

    def add(self, embeddings, metadata):
        embeddings = np.array(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        self._rebuild_bm25()

    def _vector_search(self, query_embedding, top_k):
        if query_embedding is None or self.index.ntotal == 0:
            return []

        query_embedding = np.array([query_embedding], dtype="float32")
        candidate_count = min(max(top_k, 1), self.index.ntotal)
        distances, indices = self.index.search(query_embedding, candidate_count)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            results.append({
                "idx": int(idx),
                "rank": rank,
                "vector_score": float(distances[0][rank]),
            })

        return results

    def _bm25_search(self, query_text, top_k):
        if not query_text or not self.metadata:
            return []

        scores = self.bm25.score(query_text)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[:min(max(top_k, 1), len(scores))]

        return [
            {
                "idx": int(idx),
                "rank": rank,
                "bm25_score": float(score),
            }
            for rank, (idx, score) in enumerate(ranked)
            if score > 0
        ]

    def search(self, query_embedding=None, top_k=5, query_text=None, vector_weight=0.6, bm25_weight=0.4):
        candidate_pool = max(top_k * 3, top_k)
        vector_hits = self._vector_search(query_embedding, candidate_pool)
        bm25_hits = self._bm25_search(query_text, candidate_pool)

        if query_text and bm25_hits:
            fused_scores = {}

            for hit in vector_hits:
                fused_scores.setdefault(hit["idx"], {}).update(hit)
                fused_scores[hit["idx"]]["score"] = fused_scores[hit["idx"]].get("score", 0.0) + vector_weight / (hit["rank"] + 1)

            for hit in bm25_hits:
                fused_scores.setdefault(hit["idx"], {}).update(hit)
                fused_scores[hit["idx"]]["score"] = fused_scores[hit["idx"]].get("score", 0.0) + bm25_weight / (hit["rank"] + 1)

            ranked_hits = sorted(
                fused_scores.items(),
                key=lambda item: item[1]["score"],
                reverse=True,
            )[:top_k]
        else:
            ranked_hits = [(hit["idx"], {"score": hit["vector_score"], **hit}) for hit in vector_hits[:top_k]]

        results = []
        for idx, hit in ranked_hits:
            doc = self.metadata[idx]
            results.append({
                "text": doc["text"],
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": doc.get("chunk_id"),
                "section_title": doc.get("section_title"),
                "section_type": doc.get("section_type"),
                "score": float(hit.get("score", 0.0)),
                "vector_score": hit.get("vector_score"),
                "bm25_score": hit.get("bm25_score"),
            })

        return results

    def save(self, path):
        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
