"""Embedding backends. All return L2-normalised float32 matrices (cosine == dot product)."""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np

from ..config.settings import EmbeddingConfig
from ..errors import ModelUnavailableError
from ..models.sources import resolve_model_source
from .bm25 import tokenize

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class Embedder(ABC):
    model_id: str = "unknown"
    fallback_reason: str | None = None

    @abstractmethod
    def embed_documents(self, texts: list[str], progress: ProgressFn | None = None) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...

    def close(self) -> None:  # release GPU memory etc.
        return None


class HashingEmbedder(Embedder):
    """Deterministic feature-hashing embeddings (unigrams + bigrams). Zero downloads, fully offline.
    Purely lexical, so retrieval quality is lower than a neural model; used as a safe fallback."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.model_id = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return np.sign(v) * np.log1p(np.abs(v))

    def embed_documents(self, texts: list[str], progress: ProgressFn | None = None) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text))


class OllamaEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self.base = cfg.base_url.rstrip("/")
        self.model_id = f"ollama:{cfg.model}"
        self.client = httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            r = self.client.post(f"{self.base}/api/embed", json={"model": self.cfg.model, "input": texts})
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                f"Cannot reach Ollama at {self.base}. Start it with `ollama serve`."
            ) from exc
        if r.status_code == 404:
            raise ModelUnavailableError(f"Embedding model missing. Run: ollama pull {self.cfg.model}")
        r.raise_for_status()
        return _normalize(np.array(r.json()["embeddings"], dtype=np.float32))

    def embed_documents(self, texts: list[str], progress: ProgressFn | None = None) -> np.ndarray:
        out = []
        bs = max(self.cfg.batch_size, 1)
        for i in range(0, len(texts), bs):
            out.append(self._embed(texts[i: i + bs]))
            if progress:
                progress(min((i + bs) / max(len(texts), 1), 1.0), "Creating embeddings")
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.cfg.query_prompt + text])[0]

    def close(self) -> None:
        self.client.close()


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig, models_dir: Path) -> None:
        self.cfg = cfg
        self.source = resolve_model_source(cfg.model, cfg.model_path, models_dir)
        self.model_id = f"st:{cfg.model}"
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelUnavailableError(
                "sentence-transformers/torch are not installed. Run: pip install -r requirements-transformers.txt"
            ) from exc
        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"model_kwargs": {"dtype": torch.float16}} if device == "cuda" else {}
        try:
            self._model = SentenceTransformer(self.source, device=device, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(f"Could not load embedding model '{self.source}': {exc}") from exc
        log.info("Loaded embedder %s on %s", self.source, device)

    def embed_documents(self, texts: list[str], progress: ProgressFn | None = None) -> np.ndarray:
        self.load()
        assert self._model is not None
        out = []
        bs = max(self.cfg.batch_size, 1)
        for i in range(0, len(texts), bs):
            out.append(self._model.encode(texts[i: i + bs], batch_size=bs, normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False))
            if progress:
                progress(min((i + bs) / max(len(texts), 1), 1.0), "Creating embeddings")
        return _normalize(np.concatenate(out)) if out else np.zeros((0, 1), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        self.load()
        assert self._model is not None
        v = self._model.encode([self.cfg.query_prompt + text], normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)[0]
        return _normalize(v)

    def close(self) -> None:
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def create_embedder(cfg: EmbeddingConfig, models_dir: Path, allow_fallback: bool = True) -> Embedder:
    """Build the configured embedder; fall back to lexical hashing if it cannot be loaded."""
    try:
        if cfg.provider == "hashing":
            return HashingEmbedder()
        if cfg.provider == "ollama":
            emb: Embedder = OllamaEmbedder(cfg)
            emb.embed_query("ping")  # fail fast if Ollama / model is missing
            return emb
        st = SentenceTransformerEmbedder(cfg, models_dir)
        st.load()
        return st
    except ModelUnavailableError as exc:
        if not allow_fallback:
            raise
        log.warning("Embedding backend unavailable (%s); using lexical fallback.", exc)
        fb = HashingEmbedder()
        fb.fallback_reason = str(exc)
        return fb
