"""Small SQLite database for the paper library, chat history and highlights."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers(
    doc_id TEXT PRIMARY KEY, title TEXT, source TEXT, pages INTEGER, added REAL, opened REAL);
CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, role TEXT, content TEXT,
    sources TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS idx_messages_doc ON messages(doc_id, id);
CREATE TABLE IF NOT EXISTS highlights(
    id TEXT PRIMARY KEY, doc_id TEXT, page INTEGER, rects TEXT, text TEXT,
    use_in_chat INTEGER DEFAULT 1, ts REAL);
CREATE INDEX IF NOT EXISTS idx_highlights_doc ON highlights(doc_id);
"""


class LibraryDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- papers ----
    def upsert_paper(self, doc_id: str, title: str, source: str, pages: int) -> None:
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO papers(doc_id,title,source,pages,added,opened) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(doc_id) DO UPDATE SET title=excluded.title, source=excluded.source, "
                "pages=excluded.pages, opened=excluded.opened",
                (doc_id, title, source, pages, now, now),
            )

    def recent_papers(self, limit: int = 10) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM papers ORDER BY opened DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- chat ----
    def add_message(self, doc_id: str, role: str, content: str, sources: list[dict] | None = None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages(doc_id,role,content,sources,ts) VALUES(?,?,?,?,?)",
                (doc_id, role, content, json.dumps(sources or []), time.time()),
            )

    def messages(self, doc_id: str, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role,content,sources,ts FROM messages WHERE doc_id=? ORDER BY id DESC LIMIT ?",
                (doc_id, limit),
            ).fetchall()
        return [
            {"role": r["role"], "content": r["content"], "sources": json.loads(r["sources"] or "[]")}
            for r in reversed(rows)
        ]

    def clear_messages(self, doc_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM messages WHERE doc_id=?", (doc_id,))

    # ---- highlights ----
    def save_highlight(self, doc_id: str, hid: str, page: int, rects: list, text: str, use: bool) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO highlights(id,doc_id,page,rects,text,use_in_chat,ts) "
                "VALUES(?,?,?,?,?,?,?)",
                (hid, doc_id, page, json.dumps(rects), text, int(use), time.time()),
            )

    def set_highlight_use(self, hid: str, use: bool) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE highlights SET use_in_chat=? WHERE id=?", (int(use), hid))

    def delete_highlight(self, hid: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM highlights WHERE id=?", (hid,))

    def clear_highlights(self, doc_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM highlights WHERE doc_id=?", (doc_id,))

    def highlights(self, doc_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM highlights WHERE doc_id=? ORDER BY ts", (doc_id,)
            ).fetchall()
        return [
            {"id": r["id"], "page": r["page"], "rects": json.loads(r["rects"]), "text": r["text"],
             "use_in_chat": bool(r["use_in_chat"])}
            for r in rows
        ]
