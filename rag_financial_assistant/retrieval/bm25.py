import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_lengths = []
        self.avg_doc_length = 0.0

    @staticmethod
    def tokenize(text):
        return TOKEN_RE.findall((text or "").lower())

    def add_documents(self, texts):
        for text in texts:
            tokens = self.tokenize(text)
            freqs = Counter(tokens)
            self.documents.append(tokens)
            self.doc_freqs.append(freqs)
            self.doc_lengths.append(len(tokens))

        self._recompute_idf()

    def _recompute_idf(self):
        num_docs = len(self.documents)
        if num_docs == 0:
            self.idf = {}
            self.avg_doc_length = 0.0
            return

        doc_counts = Counter()
        for tokens in self.documents:
            doc_counts.update(set(tokens))

        self.idf = {
            term: math.log(1 + (num_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_counts.items()
        }
        self.avg_doc_length = sum(self.doc_lengths) / num_docs

    def score(self, query_text):
        query_tokens = self.tokenize(query_text)
        if not query_tokens or not self.documents:
            return [0.0] * len(self.documents)

        scores = [0.0] * len(self.documents)
        for doc_id, freqs in enumerate(self.doc_freqs):
            doc_len = self.doc_lengths[doc_id] or 1
            denominator_norm = self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1))

            for token in query_tokens:
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue

                numerator = tf * (self.k1 + 1)
                denominator = tf + denominator_norm
                scores[doc_id] += self.idf.get(token, 0.0) * numerator / denominator

        return scores
