"""Two-panel main window: PDF viewer | chat, with citation navigation."""

from __future__ import annotations

from PySide6.QtWidgets import (QFileDialog, QInputDialog, QMainWindow, QSplitter,
                               QStatusBar)

from literature_buddy.config.settings import AppSettings
from literature_buddy.gui.chat_widget import ChatWidget
from literature_buddy.gui.pdf_viewer import PdfViewer
from literature_buddy.gui.workers import AnswerWorker, IndexWorker
from literature_buddy.models.backends import ModelManager
from literature_buddy.rag.memory import ConversationMemory
from literature_buddy.rag.pipeline import RAGPipeline
from literature_buddy.retrieval.embeddings import SentenceTransformerEmbeddings
from literature_buddy.retrieval.reranker import CrossEncoderReranker, IdentityReranker
from literature_buddy.retrieval.retriever import Retriever
from literature_buddy.retrieval.vector_store import VectorStore


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.setWindowTitle(settings.ui.window_title)
        self.resize(1500, 950)
        self.settings = settings
        self._doc = None
        self._pipeline = None

        self.pdf = PdfViewer(settings)
        self.chat = ChatWidget()
        splitter = QSplitter()
        splitter.addWidget(self.pdf); splitter.addWidget(self.chat)
        splitter.setSizes([700, 450])
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())

        mb = self.menuBar().addMenu("&File")
        mb.addAction("Open PDF…", self.open_pdf_dialog)
        mb.addAction("Open from URL…", self.open_url_dialog)

        self.chat.questionAsked.connect(self.on_question)
        self.chat.citationClicked.connect(self.pdf.goto_page)

        # Retrieval stack (built once; index persists on disk).
        store = VectorStore(settings.storage.resolved("index_dir"))
        embedder = SentenceTransformerEmbeddings(
            settings.embedding.model, settings.embedding.device,
            settings.embedding.batch_size)
        reranker = (CrossEncoderReranker(settings.reranker.model)
                    if settings.reranker.enabled else IdentityReranker())
        self.retriever = Retriever(store, embedder, reranker,
                                   top_k=settings.retrieval.top_k,
                                   bm25_weight=settings.retrieval.bm25_weight,
                                   rerank_top_n=settings.reranker.top_n)
        self.manager = ModelManager(settings)
        self.memory = ConversationMemory(settings.memory.max_history_turns,
                                         settings.memory.summary_threshold)

    # -- paper loading --------------------------------------------------------
    def open_pdf_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open paper", "", "PDF (*.pdf)")
        if path: self._install_paper(path)

    def open_url_dialog(self) -> None:
        url, ok = QInputDialog.getText(self, "Open URL", "Paper URL (arXiv/PMC/…)")
        if ok and url: self._install_paper(url)

    def _install_paper(self, source: str) -> None:
        self.chat.set_busy(True)
        self._worker = IndexWorker(source, self.settings, self.retriever)
        self._worker.progress.connect(self.statusBar().showMessage)
        self._worker.error.connect(self._on_index_error)
        self._worker.done.connect(self._on_paper_ready)
        self._worker.start()

    def _on_index_error(self, msg: str) -> None:
        self.chat.set_busy(False)
        self.statusBar().showMessage(f"Error: {msg}")

    def _on_paper_ready(self, doc) -> None:
        self._doc = doc
        self.pdf.open_pdf(doc.source_path)
        self._pipeline = RAGPipeline(doc, self.retriever, self.manager, self.memory,
                                     context_tokens=6_000)
        self.chat.set_busy(False)
        self.statusBar().showMessage(
            f"Loaded '{doc.title[:60]}' — {doc.num_pages} pages indexed, ready.")

    # -- Q&A ------------------------------------------------------------------
    def on_question(self, question: str) -> None:
        if self._pipeline is None:
            self.statusBar().showMessage("Open a paper first.")
            return
        self.chat.add_user_message(question)
        self.chat.set_busy(True)
        self._ans_worker = AnswerWorker(self._pipeline, question)
        self._ans_worker.finished_answer.connect(self._on_answer)
        self._ans_worker.error.connect(self._on_index_error)
        self._ans_worker.start()

    def _on_answer(self, text: str, citations) -> None:
        self.chat.start_assistant_message()
        self.chat.append_assistant_chunk(text)
        html = "<br>".join(c.as_html() for c in citations)
        self.chat.finish_assistant_message(html)
        self.chat.set_busy(False)