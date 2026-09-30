"""On-disk layout:

    <data_dir>/
        papers/<doc_id>/source.pdf, document.json, index.sqlite, figures/*.png
        downloads/            (temporary downloaded PDFs)
        library.sqlite        (recent papers, chat history, highlights)
        logs/
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..document.loader import sha256_file


def short_hash(path: Path) -> str:
    return sha256_file(path)[:16]


class PaperCache:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def paper_dir(self, doc_id: str) -> Path:
        d = self.root / "papers" / doc_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def downloads_dir(self) -> Path:
        d = self.root / "downloads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def library_path(self) -> Path:
        return self.root / "library.sqlite"

    def store_pdf(self, src: Path, doc_id: str) -> Path:
        dst = self.paper_dir(doc_id) / "source.pdf"
        if not dst.exists():
            shutil.copy2(src, dst)
        return dst

    def purge(self, doc_id: str) -> None:
        shutil.rmtree(self.root / "papers" / doc_id, ignore_errors=True)
