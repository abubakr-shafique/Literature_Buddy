# src/literature_buddy/retrieval/vector_store.py
"""Persistent local vector store backed by ChromaDB.

Chosen over FAISS for: persistent by default, metadata filtering built-in,
zero server process needed, and fully offline. One collection per paper
(slug) so indexes are reusable across sessions; a paper is embedded once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from literature_buddy.document.models import Chunk


class VectorStore:
    def __init__(self, index_dir: str | Path):
        index_dir = Path(index_dir).expanduser()
        index_dir.mkdir(parents=True, exist_ok=True)
        import chromadb

        self._client = chromadb.PersistentClient(path=str(index_dir))

    def has_paper(self, slug: str) -> bool:
        try:
            col = self._client.get_collection(slug)
            return col.count() > 0
        except Exception:
            return False

    def _collection(self, slug: str):
        return self._client.get_or_create_collection(
            name=slug, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, slug: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        col = self._collection(slug)
        col.add(
            ids=[f"{slug}-{i}" for i in range(len(chunks))],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{
                "page": c.page,
                "section": c.section,
                "section_title": c.section_title,
                "chunk_type": c.chunk_type,
                "figure_label": c.figure_label,
                "table_label": c.table_label,
                "figure_image_path": c.figure_image_path,
            } for c in chunks],
        )

    def query(
        self, slug: str, query_vec: np.ndarray, top_k: int
    ) -> tuple[list[str], list[dict], list[float]]:
        col = self._collection(slug)
        if col.count() == 0:
            return [], [], []
        res = col.query(query_embeddings=[query_vec.tolist()], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        scores = [1.0 - d for d in dists]  # cosine distance -> similarity
        return docs, metas, scores
