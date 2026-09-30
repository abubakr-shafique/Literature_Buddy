"""Application services shared by the GUI and CLI (config-dependent, lazily initialised)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from .config.settings import AppConfig
from .document.loader import download_pdf, is_url
from .document.processor import DocumentProcessor, ProcessedPaper
from .models.model_manager import ModelManager
from .models.sources import apply_offline_env
from .rag.memory import ConversationMemory
from .rag.pipeline import RagPipeline
from .retrieval.embeddings import Embedder, create_embedder
from .retrieval.reranker import Reranker, create_reranker
from .retrieval.retriever import HybridRetriever
from .storage import LibraryDB, PaperCache

log = logging.getLogger(__name__)


class Services:
    def __init__(self, cfg: AppConfig) -> None:
        apply_offline_env(cfg.offline)
        self.cfg = cfg
        self.cache = PaperCache(cfg.data_path)
        self.db = LibraryDB(self.cache.library_path)
        self.models = ModelManager(cfg)
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None
        self._reranker_tried = False
        self._lock = threading.Lock()

    @property
    def embedder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                self._embedder = create_embedder(self.cfg.embedding, self.cfg.models_path)
            return self._embedder

    @property
    def reranker(self) -> Reranker | None:
        with self._lock:
            if not self._reranker_tried:
                self._reranker_tried = True
                self._reranker = create_reranker(self.cfg.retrieval, self.cfg.models_path, self.cfg.embedding.device)
            return self._reranker

    def load_paper(self, source: str, progress=None, cancel: threading.Event | None = None) -> ProcessedPaper:
        """Local path or URL -> processed, indexed paper."""
        if is_url(source):
            path = download_pdf(source, self.cache.downloads_dir, self.cfg.document.max_download_mb,
                                lambda f, m: progress and progress(f * 0.15, m), cancel)
            base = 0.15
        else:
            path, base = Path(source).expanduser(), 0.0
        emb = self.embedder
        proc = DocumentProcessor(self.cfg, emb, self.cache)
        paper = proc.process(path, source, lambda f, m: progress and progress(base + f * (1 - base), m), cancel)
        self.db.upsert_paper(paper.document.doc_id, paper.document.title, source, paper.document.page_count)
        return paper

    def pipeline_for(self, paper: ProcessedPaper) -> RagPipeline:
        retriever = HybridRetriever(paper.store, self.embedder, self.reranker, self.cfg.retrieval, paper.document.title)
        memory = ConversationMemory(self.cfg.conversation.history_tokens)
        for m in self.db.messages(paper.document.doc_id, limit=12):
            if m["role"] in ("user", "assistant"):
                memory.add(m["role"], m["content"])
        return RagPipeline(self.cfg, paper, retriever, self.models, memory)

    def shutdown(self) -> None:
        self.models.unload_all()
        if self._embedder:
            self._embedder.close()
        self.db.close()
