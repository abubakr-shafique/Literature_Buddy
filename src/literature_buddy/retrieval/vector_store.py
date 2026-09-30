"""Per-paper index in a single SQLite file (chunks + float32 vectors + small key/value tables).

Why not FAISS/Chroma/Qdrant/LanceDB? One paper is a few hundred chunks: exact cosine search over an
in-memory numpy matrix takes microseconds, the index is one portable file, and there are no extra
dependencies. The `VectorStore` surface is small so a LanceDB/Qdrant backend can replace this once
multi-paper libraries are supported (see docs/architecture.md).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

from ..document.schema import Chunk
from .bm25 import BM25, tokenize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY, ord INTEGER, json TEXT NOT NULL, vec BLOB);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
"""


class SqliteVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._bm25: BM25 | None = None
        self._by_id: dict[str, Chunk] = {}
        self.load()

    # ---- persistence ----
    def load(self) -> None:
        with self._lock:
            rows = self._db.execute("SELECT json, vec FROM chunks ORDER BY ord").fetchall()
        self.chunks = [Chunk.model_validate_json(r[0]) for r in rows]
        vecs = [np.frombuffer(r[1], dtype=np.float32) for r in rows if r[1] is not None]
        self._matrix = np.stack(vecs) if len(vecs) == len(rows) and vecs else None
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._bm25 = BM25([tokenize(f"{c.section} {c.label or ''} {c.text}") for c in self.chunks])

    def replace_all(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict[str, str]) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks")
            self._db.executemany(
                "INSERT INTO chunks(chunk_id, ord, json, vec) VALUES (?,?,?,?)",
                [(c.chunk_id, c.ord, c.model_dump_json(), vecs[i].tobytes()) for i, c in enumerate(chunks)],
            )
            for k, v in meta.items():
                self._db.execute("INSERT OR REPLACE INTO kv(key,value) VALUES(?,?)", (k, v))
        self.load()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO kv(key,value) VALUES(?,?)", (key, value))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- search ----
    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def dense_scores(self, qvec: np.ndarray) -> np.ndarray:
        if self._matrix is None or not len(self.chunks):
            return np.zeros(len(self.chunks), dtype=np.float32)
        return self._matrix @ np.asarray(qvec, dtype=np.float32)

    def sparse_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return self._bm25.scores(tokenize(query))

    def by_label(self, label: str) -> list[Chunk]:
        return [c for c in self.chunks if c.label and c.label.lower() == label.lower()]
