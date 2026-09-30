"""Ingestion pipeline: PDF -> Document -> chunks -> embeddings -> persistent index (with caching)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config.settings import AppConfig
from ..errors import Cancelled
from ..retrieval.embeddings import Embedder
from ..retrieval.vector_store import SqliteVectorStore
from ..storage.cache import PaperCache, short_hash
from .chunker import build_chunks
from .loader import validate_pdf
from .parser import build_key, parse_pdf
from .schema import SCHEMA_VERSION, Document

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]


@dataclass
class ProcessedPaper:
    document: Document
    store: SqliteVectorStore
    directory: Path
    pdf_path: Path
    from_cache: bool = False

    def image_path(self, rel: str | None) -> Path | None:
        return (self.directory / rel) if rel else None


class DocumentProcessor:
    def __init__(self, cfg: AppConfig, embedder: Embedder, cache: PaperCache) -> None:
        self.cfg, self.embedder, self.cache = cfg, embedder, cache

    def process(
        self, pdf: Path, source: str | None = None, progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
    ) -> ProcessedPaper:
        def report(f: float, m: str) -> None:
            if progress:
                progress(f, m)

        def check() -> None:
            if cancel is not None and cancel.is_set():
                raise Cancelled("Cancelled by user")

        report(0.0, "Checking file")
        validate_pdf(pdf)
        doc_id = short_hash(pdf)
        stored = self.cache.store_pdf(pdf, doc_id)
        pdir = self.cache.paper_dir(doc_id)
        json_path = pdir / "document.json"
        key = build_key(self.cfg.document)

        document: Document | None = None
        if json_path.exists():
            try:
                cached = Document.model_validate_json(json_path.read_text(encoding="utf-8"))
                if cached.build_key == key and cached.schema_version == SCHEMA_VERSION:
                    document = cached
            except Exception:  # noqa: BLE001 - corrupt cache: rebuild
                log.warning("Discarding unreadable cache for %s", doc_id)
        parsed_now = document is None
        if document is None:
            document = parse_pdf(stored, doc_id, source or str(pdf), pdir, self.cfg.document,
                                 lambda f, m: report(f * 0.75, m), cancel)
            json_path.write_text(document.model_dump_json(indent=1), encoding="utf-8")
        check()

        store = SqliteVectorStore(pdir / "index.sqlite")
        chunks = build_chunks(document, self.cfg.document)
        fingerprint = f"{self.embedder.model_id}|{key}|{len(chunks)}"
        if not parsed_now and store.get_meta("fingerprint") == fingerprint and len(store.chunks) == len(chunks):
            report(1.0, "Ready (cached)")
            return ProcessedPaper(document, store, pdir, stored, from_cache=True)

        report(0.78, "Creating embeddings")
        texts = [c.embedding_text(document.title) for c in chunks]
        vectors = self.embedder.embed_documents(texts, lambda f, m: report(0.78 + 0.2 * f, m))
        check()
        store.replace_all(chunks, vectors, {"fingerprint": fingerprint, "embedder": self.embedder.model_id})
        report(1.0, "Ready")
        return ProcessedPaper(document, store, pdir, stored)
