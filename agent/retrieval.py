"""
agent/retrieval.py — Hybrid retrieval: FAISS (dense) + BM25 (sparse) + RRF fusion.
Same proven pattern from document-agent.
"""

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def build(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([self._tokenize(c) for c in chunks])

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(self._tokenize(query))
        top = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top]


class FAISSRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks: list[str]) -> None:
        self.chunks = chunks
        embeddings = self.model.encode(chunks, show_progress_bar=False).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(q, top_k)
        return [(int(indices[0][i]), float(distances[0][i])) for i in range(top_k)]


def _rrf(bm25_results, faiss_results, k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(faiss_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class VectorStore:
    """Hybrid retrieval: FAISS + BM25 + RRF."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.faiss = FAISSRetriever(model_name)
        self.bm25 = BM25Retriever()
        self.chunks = []

    def build(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.faiss.build(chunks)
        self.bm25.build(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self.chunks:
            raise ValueError("Index not built. Call build() first.")
        candidate_k = min(top_k * 3, len(self.chunks))
        fused = _rrf(
            self.bm25.retrieve(query, candidate_k),
            self.faiss.retrieve(query, candidate_k),
        )
        return [self.chunks[idx] for idx, _ in fused[:top_k] if idx < len(self.chunks)]

    @property
    def size(self) -> int:
        return len(self.chunks)