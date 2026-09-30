"""Optional cross-encoder reranking (BAAI/bge-reranker-v2-m3 by default)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..config.settings import RetrievalConfig
from ..models.sources import resolve_model_source

log = logging.getLogger(__name__)


class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker(Reranker):
    def __init__(self, cfg: RetrievalConfig, models_dir: Path, device: str = "auto") -> None:
        import torch
        from sentence_transformers import CrossEncoder

        source = resolve_model_source(cfg.reranker_model, cfg.reranker_path, models_dir)
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self._model = CrossEncoder(source, device=dev, max_length=512)
        log.info("Loaded reranker %s on %s", source, dev)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        preds = self._model.predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False)
        return [float(x) for x in preds]


def create_reranker(cfg: RetrievalConfig, models_dir: Path, device: str = "auto") -> Reranker | None:
    if not cfg.rerank:
        return None
    try:
        return CrossEncoderReranker(cfg, models_dir, device)
    except Exception as exc:  # noqa: BLE001 - missing deps / weights: degrade gracefully
        log.warning("Reranker disabled (%s). Hybrid search still works.", exc)
        return None
