"""Hybrid retrieval: dense (embeddings) + sparse (BM25) -> reciprocal-rank fusion -> optional rerank."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config.settings import RetrievalConfig
from ..document.schema import Chunk
from .embeddings import Embedder
from .reranker import Reranker
from .vector_store import SqliteVectorStore

log = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


class HybridRetriever:
    def __init__(
        self, store: SqliteVectorStore, embedder: Embedder, reranker: Reranker | None,
        cfg: RetrievalConfig, title: str = "",
    ) -> None:
        self.store, self.embedder, self.reranker, self.cfg, self.title = store, embedder, reranker, cfg, title

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        kinds: set[str] | None = None,
        exclude_kinds: set[str] | None = None,
        prefer_sections: set[str] | None = None,
    ) -> list[Hit]:
        chunks = self.store.chunks
        if not chunks:
            return []
        k = top_k or self.cfg.top_k
        pool = self.cfg.candidates
        allowed = np.array(
            [
                (kinds is None or c.kind in kinds) and (not exclude_kinds or c.kind not in exclude_kinds)
                for c in chunks
            ]
        )
        if not allowed.any():
            return []

        dense = self.store.dense_scores(self.embedder.embed_query(query))
        dense = np.where(allowed, dense, -np.inf)
        dense_order = [int(i) for i in np.argsort(-dense)[:pool] if np.isfinite(dense[i])]

        sparse_order: list[int] = []
        if self.cfg.hybrid:
            sparse = np.where(allowed, self.store.sparse_scores(query), 0.0)
            sparse_order = [int(i) for i in np.argsort(-sparse)[:pool] if sparse[i] > 0]

        fused: dict[int, Hit] = {}
        for rank, i in enumerate(dense_order):
            fused[i] = Hit(chunks[i], 1.0 / (self.cfg.rrf_k + rank + 1), dense_rank=rank)
        for rank, i in enumerate(sparse_order):
            add = 1.0 / (self.cfg.rrf_k + rank + 1)
            if i in fused:
                fused[i].score += add
                fused[i].sparse_rank = rank
            else:
                fused[i] = Hit(chunks[i], add, sparse_rank=rank)
        if prefer_sections:
            for h in fused.values():
                if h.chunk.section_kind in prefer_sections:
                    h.score *= 1.25
        hits = sorted(fused.values(), key=lambda h: -h.score)

        if self.reranker is not None and hits:
            head = hits[: max(k * 2, 12)]
            try:
                scores = self.reranker.score(query, [h.chunk.embedding_text(self.title) for h in head])
                for h, s in zip(head, scores, strict=True):
                    h.rerank_score = s
                head.sort(key=lambda h: -(h.rerank_score or 0.0))
                hits = head + hits[len(head):]
            except Exception as exc:  # noqa: BLE001
                log.warning("Reranking failed (%s); using fused order.", exc)
        return hits[:k]
