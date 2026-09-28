# src/literature_buddy/retrieval/reranker.py
"""Cross-encoder reranking.

Default: BAAI/bge-reranker-v2-m3 (568M params, multilingual, runs comfortably
on any modern GPU or CPU). Rerankers are applied to the retrieval shortlist,
not the full index, keeping latency practical.
"""

from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list, top_n: int | None = None) -> list:
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self._model.predict(pairs)
        for h, s in zip(hits, scores):
            h.score = float(s)
        ranked = sorted(hits, key=lambda h: h.score, reverse=True)
        return ranked[: top_n or len(ranked)]


class IdentityReranker:
    """No-op; used when reranking is disabled and in tests."""

    def rerank(self, query: str, hits: list, top_n: int | None = None) -> list:
        return hits[: top_n or len(hits)]
