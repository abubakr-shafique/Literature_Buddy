"""QThread workers: GUI never blocks during parse/index/generate (spec §19)."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from literature_buddy.document.chunker import chunk_document
from literature_buddy.document.loader import load_paper
from literature_buddy.document.parser import parse_pdf


class IndexWorker(QThread):
    progress = Signal(str)                 # human-readable status lines
    done = Signal(object)                  # ParsedDocument
    error = Signal(str)

    def __init__(self, source: str, settings, retriever):
        super().__init__()
        self._source, self._settings, self._retriever = source, settings, retriever

    def run(self) -> None:
        try:
            self.progress.emit("Downloading / loading paper…")
            path = load_paper(self._source, self._settings.storage.resolved("papers_dir"))
            self.progress.emit("Parsing paper (sections, figures, tables)…")
            doc = parse_pdf(path, self._settings)
            self.progress.emit(f"Parsed {doc.num_pages} pages, "
                               f"{len(doc.figures)} figures, {len(doc.sections)} sections")
            self.progress.emit("Chunking + building retrieval index…")
            chunks = chunk_document(doc)
            self._retriever.index_paper(
                doc.slug, chunks,
                on_progress=lambda d, t: self.progress.emit(f"Embedding {d}/{t}…"))
            self.progress.emit("Ready.")
            self.done.emit(doc)
        except Exception as exc:  # noqa: BLE001 — surface to GUI status bar
            self.error.emit(str(exc))


class AnswerWorker(QThread):
    token = Signal(str)
    finished_answer = Signal(str, list)    # answer text, [Citation]
    error = Signal(str)

    def __init__(self, pipeline, question: str):
        super().__init__()
        self._pipeline, self._q = pipeline, question

    def run(self) -> None:
        try:
            answer, cits = self._pipeline.answer(self._q)
            self.finished_answer.emit(answer, cits)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))