# src/literature_buddy/retrieval/retriever.py
"""Hybrid multimodal retrieval engine.

Dense (BGE-M3 embeddings in ChromaDB) + BM25 keyword retrieval with weighted
score fusion, a figure-label regex boost (a query mentioning "Figure 3"
directly promotes that figure's chunk), then cross-encoder reranking.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from literature_buddy.document.models import Chunk
from literature_buddy.retrieval.embeddings import EmbeddingBackend
from literature_buddy.retrieval.vector_store import VectorStore

FIGURE_Q_RE = re.compile(r"\bfig(?:ure)?\.?\s*(\d+[a-z]?)", re.IGNORECASE)
TABLE_Q_RE = re.compile(r"\btable\s*(\d+[a-z]?)", re.IGNORECASE)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float

    @property
    def is_visual(self) -> bool:
        return self.chunk.chunk_type == "figure_caption" and bool(self.chunk.figure_image_path)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingBackend,
        reranker=None,
        top_k: int = 12,
        bm25_weight: float = 0.3,
        rerank_top_n: int = 6,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.rerank_top_n = rerank_top_n
        self._bm25: dict[str, tuple[BM25Okapi, list[Chunk]]] = {}

    # -- indexing -----------------------------------------------------------
    def index_paper(
        self,
        slug: str,
        chunks: list[Chunk],
        on_progress=None,
    ) -> None:
        if self.store.has_paper(slug):
            self._load_bm25(slug, chunks)
            return
        all_vecs = []
        batch = 64
        for i in range(0, len(chunks), batch):
            part = self.embedder.embed([c.text for c in chunks[i:i + batch]])
            all_vecs.append(part)
            if on_progress:
                on_progress(min(i + batch, len(chunks)), len(chunks))
        vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0, 1), dtype=np.float32)
        self.store.add_chunks(slug, chunks, vecs)
        self._load_bm25(slug, chunks)

    def _load_bm25(self, slug: str, chunks: list[Chunk]) -> None:
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25[slug] = (BM25Okapi(tokenized), chunks)

    # -- querying -----------------------------------------------------------
    def retrieve(self, slug: str, query: str) -> RetrievalResult:
        result = RetrievalResult(query=query)
        q_vec = self.embedder.embed([query])[0]
        docs, metas, dense_scores = self.store.query(slug, q_vec, self.top_k)
        dense_hits = {
            m.get("figure_label") or f"doc-{i}": (d, m, s)
            for i, (d, m, s) in enumerate(zip(docs, metas, dense_scores))
        }

        bm = self._bm25.get(slug)
        fused: dict[str, RetrievedChunk] = {}
        text_docs, text_metas, text_scores = [], [], []
        if bm:
            model, all_chunks = bm
            bm_scores = model.get_scores(query.lower().split())
            norm = _minmax(bm_scores)
            order = np.argsort(bm_scores)[::-1][: self.top_k * 2]
            for idx in order:
                d = all_chunks[int(idx)]
                fused_chunk = RetrievedChunk(
                    chunk=d,
                    score=self.bm25_weight * float(norm[int(idx)]),
                )
                fused[f"bm-{idx}"] = fused_chunk

        fig_boost = _figure_labels_from_query(query)
        tbl_boost = _table_labels_from_query(query)

        for i, (d, m, s) in enumerate(zip(docs, metas, dense_scores)):
            chunk = Chunk(
                text=d,
                page=int(m.get("page", 1)),
                section=m.get("section", "other"),
                section_title=m.get("section_title", ""),
                chunk_type=m.get("chunk_type", "text"),
                figure_label=m.get("figure_label", ""),
                table_label=m.get("table_label", ""),
                figure_image_path=m.get("figure_image_path", ""),
            )
            score = (1.0 - self.bm25_weight) * float(s)
            if chunk.figure_label.lower() in fig_boost or chunk.table_label.lower() in tbl_boost:
                score += 0.5
            key = f"dense-{i}"
            if key in fused:
                fused[key].score += score
            else:
                fused[key] = RetrievedChunk(chunk=chunk, score=score)

        hits = sorted(fused.values(), key=lambda r: r.score, reverse=True)[: self.top_k]
        if self.reranker is not None and hits:
            hits = self.reranker.rerank(query, hits)[: self.rerank_top_n]

        result.chunks = hits
        return result


def _minmax(xs: np.ndarray) -> np.ndarray:
    lo, hi = float(xs.min()), float(xs.max())
    if math.isclose(lo, hi):
        return np.zeros_like(xs)
    return (xs - lo) / (hi - lo)


def _figure_labels_from_query(q: str) -> set[str]:
    return {f"figure {m.group(1)}".lower() for m in FIGURE_Q_RE.finditer(q)}


def _table_labels_from_query(q: str) -> set[str]:
    return {f"table {m.group(1)}".lower() for m in TABLE_Q_RE.finditer(q)}
