# src/literature_buddy/retrieval/embeddings.py
"""Embedding backend (sentence-transformers).

Default: BAAI/bge-m3 — strong on scientific retrieval benchmarks, multilingual,
8192-token context, ~2.3 GB. Runs on tiny GPU RAM; CPU fallback is viable.
SPECTER2 is offered in docs for paper-level similarity tasks.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32):
        import torch  # local import: keeps module importable without torch

        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        vec = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vec, dtype=np.float32)


class HashEmbeddings:
    """Deterministic fallback for tests/offline dev (no model download)."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, hash(tok) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
