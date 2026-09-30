"""Main window: PDF viewer | chat + highlights, toolbar, menus, background tasks."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThreadPool
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QToolBar,
    QWidget,
)

from .. import __version__
from ..config.settings import save_user_config
from ..document.loader import is_url
from ..errors import LiteratureBuddyError
from ..rag.pipeline import AnswerResult, RagPipeline
from ..services import Services
from .chat_widget import ChatWidget
from .highlights_panel import HighlightsPanel
from .pdf_viewer import PdfViewer
from .settings_dialog import SettingsDialog
from .tasks import Task
from .types import Highlight

log = logging.getLogger(__name__)
LOGO = Path(__file__).parent.parent / "resources" / "logo.svg"


class MainWindow(QMainWindow):
    def __init__(self, services: Services, initial: str | None = None) -> None:
        super().__init__()
        self.services = services
        self.paper = None
        self.pipeline: RagPipeline | None = None
        self._tasks: set[Task] = set()
        self._ask_task: Task | None = None
        self._load_task: Task | None = None
        self._external_ok = False
        self._last_search = ""

        self.setWindowTitle("Literature Buddy")
        self.setWindowIcon(QIcon(str(LOGO)))
        self.resize(1440, 900)
        self.setAcceptDrops(True)

        self.viewer = PdfViewer()
        self.chat = ChatWidget()
        self.highlights = HighlightsPanel()
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.chat)
        right.addWidget(self.highlights)
        right.setStretchFactor(0, 5)
        right.setStretchFactor(1, 1)
        right.setSizes([650, 190])
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.viewer)
        split.addWidget(right)
        split.setStretchFactor(0, 6)
        split.setStretchFactor(1, 4)
        split.setSizes([860, 580])
        self.setCentralWidget(split)

        self._build_actions()
        self._build_status()
        self._connect()
        self._refresh_recent()
        self._check_backend()
        self.chat.add_notice("<b>Welcome to Literature Buddy.</b><br>Open a paper (Ctrl+O), paste a URL (Ctrl+L), "
                             "or drop a PDF on the left panel. Answers are grounded in the paper and cite pages, "
                             "figures and tables.")
        if initial:
            self.open_source(initial)

    # ------------------------------------------------------------------ UI construction
    def _act(self, text: str, slot, shortcut: str | QKeySequence.StandardKey | None = None, tip: str = "") -> QAction:
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
        return a

    def _build_actions(self) -> None:
        self.a_open = self._act("📂 Open PDF", self.open_dialog, "Ctrl+O", "Open a local PDF (Ctrl+O)")
        self.a_url = self._act("🔗 Open URL", self.open_url_dialog, "Ctrl+L", "Open an open-access paper by URL (Ctrl+L)")
        self.a_prev = self._act("◀", self.viewer.prev_page, tip="Previous page")
        self.a_next = self._act("▶", self.viewer.next_page, tip="Next page")
        self.a_zin = self._act("＋", self.viewer.zoom_in, "Ctrl+=", "Zoom in")
        self.a_zout = self._act("－", self.viewer.zoom_out, "Ctrl+-", "Zoom out")
        self.a_fit = self._act("Fit width", self.viewer.fit_width, "Ctrl+0")
        self.a_settings = self._act("⚙ Settings", self.open_settings, tip="Models and retrieval settings")

        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)
        for a in (self.a_open, self.a_url):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.a_prev)
        self.page_spin = QSpinBox()
        self.page_spin.setRange(1, 1)
        self.page_spin.setFixedWidth(70)
        self.page_total = QLabel(" / –")
        tb.addWidget(self.page_spin)
        tb.addWidget(self.page_total)
        tb.addAction(self.a_next)
        tb.addSeparator()
        for a in (self.a_zout, self.a_zin, self.a_fit):
            tb.addAction(a)
        tb.addSeparator()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search in paper (Ctrl+F)")
        self.search_box.setFixedWidth(220)
        self.search_box.setClearButtonEnabled(True)
        tb.addWidget(self.search_box)
        tb.addAction(self._act("↑", lambda: self.viewer.find_step(-1), tip="Previous match"))
        tb.addAction(self._act("↓", lambda: self.viewer.find_step(1), tip="Next match"))
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding, spacer.sizePolicy().verticalPolicy().Preferred)
        tb.addWidget(spacer)
        tb.addAction(self.a_settings)
        self.addAction(self._act("Find", lambda: self.search_box.setFocus(), "Ctrl+F"))

        m = self.menuBar()
        f = m.addMenu("&File")
        f.addAction(self.a_open)
        f.addAction(self.a_url)
        self.recent_menu = f.addMenu("Recent papers")
        f.addSeparator()
        f.addAction(self._act("Export chat as Markdown…", self.export_chat))
        f.addSeparator()
        f.addAction(self._act("Quit", self.close, "Ctrl+Q"))
        v = m.addMenu("&View")
        for a in (self.a_zin, self.a_zout, self.a_fit):
            v.addAction(a)
        c = m.addMenu("&Chat")
        c.addAction(self._act("Clear chat", self.clear_chat))
        mo = m.addMenu("&Models")
        mo.addAction(self.a_settings)
        mo.addAction(self._act("Check backend", self._check_backend))
        mo.addAction(self._act("Unload models from memory", self._unload))
        h = m.addMenu("&Help")
        h.addAction(self._act("About", self.about))

    def _build_status(self) -> None:
        sb = self.statusBar()
        self.status_lbl = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(200)
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.backend_lbl = QLabel("Model: checking…")
        self.cloud_lbl = QLabel("")
        self.cloud_lbl.setStyleSheet("color:#dc2626; font-weight:600;")
        sb.addWidget(self.status_lbl, 1)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.cloud_lbl)
        sb.addPermanentWidget(self.backend_lbl)
        hosts = self.services.models.external_hosts()
        if hosts:
            self.cloud_lbl.setText("☁ data leaves this machine → " + ", ".join(hosts))

    def _connect(self) -> None:
        v = self.viewer
        v.openRequested.connect(self.open_dialog)
        v.fileDropped.connect(self.open_source)
        v.urlDropped.connect(self.open_source)
        v.selected.connect(self._on_selection)
        v.highlightRemoveRequested.connect(lambda hid: self._remove_highlights([hid]))
        v.pageChanged.connect(self._on_page)
        self.page_spin.valueChanged.connect(lambda n: self.viewer.goto(n - 1))
        self.search_box.returnPressed.connect(self._do_search)
        self.highlights.toggled.connect(self._on_toggle)
        self.highlights.removed.connect(self._remove_highlights)
        self.highlights.activated.connect(self._goto_highlight)
        self.chat.sendRequested.connect(self._on_send)
        self.chat.stopRequested.connect(self._on_stop)
        self.chat.citationClicked.connect(self._on_cite)

    # ------------------------------------------------------------------ task helper
    def _run(self, fn, on_result, on_error=None, on_progress=None, on_finished=None) -> Task:
        t = Task(fn)
        t.signals.result.connect(on_result)
        t.signals.error.connect(on_error or self._show_error)
        if on_progress:
            t.signals.progress.connect(on_progress)
        t.signals.finished.connect(lambda: (self._tasks.discard(t), on_finished() if on_finished else None))
        self._tasks.add(t)
        QThreadPool.globalInstance().start(t)
        return t

    def _show_error(self, exc: object) -> None:
        msg = str(exc) if isinstance(exc, LiteratureBuddyError) else f"{type(exc).__name__}: {exc}"
        self.chat.add_notice(f"<b>Problem:</b> {msg.replace('<', '&lt;')}", "error")
        self.status_lbl.setText("Error")

    # ------------------------------------------------------------------ opening papers
    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open scientific paper", "", "PDF files (*.pdf)")
        if path:
            self.open_source(path)

    def open_url_dialog(self) -> None:
        url, ok = QInputDialog.getText(self, "Open URL", "Link to an open-access paper (arXiv, bioRxiv, PMC, PDF…):")
        if ok and url.strip():
            self.open_source(url.strip())

    def open_source(self, source: str) -> None:
        if self._load_task is not None:
            QMessageBox.information(self, "Busy", "A paper is already being processed.")
            return
        if not is_url(source) and not Path(source).expanduser().exists():
            QMessageBox.warning(self, "File not found", source)
            return
        self.progress.setValue(0)
        self.progress.show()
        self.status_lbl.setText("Opening paper…")
        self.chat.set_enabled(False)
        svc = self.services

        def fn(task: Task):
            paper = svc.load_paper(source, lambda f, m: task.signals.progress.emit(f, m), task.cancel)
            task.signals.progress.emit(0.99, "Preparing retrieval")
            return paper, svc.pipeline_for(paper)

        def finished() -> None:
            self._load_task = None
            self.progress.hide()

        self._load_task = self._run(
            fn, self._on_loaded, on_error=lambda e: (self._show_error(e), self.chat.set_enabled(self.paper is not None)),
            on_progress=lambda f, m: (self.progress.setValue(int(f * 100)), self.status_lbl.setText(m + "…")),
            on_finished=finished)

    def _on_loaded(self, result) -> None:
        paper, pipeline = result
        self.paper, self.pipeline = paper, pipeline
        doc = paper.document
        try:
            self.viewer.open(str(paper.pdf_path))
        except Exception as exc:  # noqa: BLE001
            self._show_error(exc)
            return
        self.setWindowTitle(f"{doc.title[:80]} — Literature Buddy")
        self.page_spin.blockSignals(True)
        self.page_spin.setRange(1, doc.page_count)
        self.page_spin.setValue(1)
        self.page_spin.blockSignals(False)
        self.page_total.setText(f" / {doc.page_count}")

        hl = [Highlight(h["id"], h["page"], [tuple(r) for r in h["rects"]], h["text"], h["use_in_chat"])
              for h in self.services.db.highlights(doc.doc_id)]
        self.viewer.set_highlights(hl)
        self.highlights.set_items(hl)

        self.chat.clear()
        self.chat.set_image_root(paper.directory)
        history = self.services.db.messages(doc.doc_id)
        def n(k: int, word: str) -> str:
            return f"{k} {word}{'' if k == 1 else 's'}"

        info = (f"{n(doc.page_count, 'page')} · {n(len(doc.sections) - 1, 'section')} · {n(len(doc.figures), 'figure')} · "
                f"{n(len(doc.tables), 'table')} · {n(len(doc.references), 'reference')}"
                + (" · loaded from cache" if paper.from_cache else ""))
        if history:
            self.chat.load_history(history)
            self.chat.add_notice(f"<b>Restored</b> your previous conversation. {info}")
        else:
            self.chat.add_welcome(doc.title, info)
        for w in doc.warnings:
            self.chat.add_notice(f"⚠ {w}")
        fb = self.services.embedder.fallback_reason
        if fb:
            self.chat.add_notice(f"⚠ Neural embeddings unavailable, using a lexical fallback: {fb}", "error")
        self.chat.set_enabled(True)
        self.status_lbl.setText("Ready")
        self._refresh_recent()

    def _refresh_recent(self) -> None:
        self.recent_menu.clear()
        for p in self.services.db.recent_papers(8):
            src = p["source"]
            if is_url(src) or Path(src).exists():
                self.recent_menu.addAction((p["title"] or src)[:70], lambda s=src: self.open_source(s))
        self.recent_menu.setEnabled(not self.recent_menu.isEmpty())

    # ------------------------------------------------------------------ viewer interactions
    def _on_page(self, page: int) -> None:
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(page + 1)
        self.page_spin.blockSignals(False)

    def _do_search(self) -> None:
        text = self.search_box.text().strip()
        if text == self._last_search and text:
            self.viewer.find_step(1)
            return
        self._last_search = text
        n = self.viewer.search(text)
        self.status_lbl.setText(f"{n} match(es) for “{text}”" if text else "Ready")

    def _on_selection(self, page: int, rects: list, text: str) -> None:
        if self.paper is None:
            return
        h = Highlight(self.viewer.new_id(), page, [tuple(r) for r in rects], text, True)
        self.viewer.add_highlight(h)
        self.highlights.add(h)
        self.services.db.save_highlight(self.paper.document.doc_id, h.hid, page, [list(r) for r in rects], text, True)
        self.status_lbl.setText("Highlighted — ticked as chat context (untick it in the list to ignore it)")

    def _on_toggle(self, hid: str, use: bool) -> None:
        for h in self.viewer.highlights:
            if h.hid == hid:
                h.use_in_chat = use
        self.viewer.refresh()
        self.services.db.set_highlight_use(hid, use)

    def _remove_highlights(self, hids: list[str]) -> None:
        self.viewer.set_highlights([h for h in self.viewer.highlights if h.hid not in hids])
        self.highlights.remove(hids)
        for hid in hids:
            self.services.db.delete_highlight(hid)

    def _goto_highlight(self, hid: str) -> None:
        for h in self.viewer.highlights:
            if h.hid == hid:
                self.viewer.goto(h.page, h.rects[0] if h.rects else None)

    # ------------------------------------------------------------------ chat
    def _confirm_external(self) -> bool:
        hosts = self.services.models.external_hosts()
        if not hosts or self._external_ok:
            return True
        r = QMessageBox.question(
            self, "Data leaves this computer",
            "Your question and excerpts of this paper will be sent to:\n\n" + "\n".join(hosts) +
            "\n\nContinue?")
        self._external_ok = r == QMessageBox.StandardButton.Yes
        return self._external_ok

    def _on_send(self, question: str) -> None:
        if self.pipeline is None or self.paper is None or not self._confirm_external():
            return
        doc_id = self.paper.document.doc_id
        self.services.db.add_message(doc_id, "user", question)
        self.chat.add_user(question)
        self.chat.begin_assistant()
        self.chat.set_busy(True, "Working…")
        pipeline, hl = self.pipeline, self.highlights.context()

        def fn(task: Task):
            final = None
            for ev in pipeline.ask(question, hl, task.cancel):
                if ev.kind == "status":
                    task.signals.status.emit(ev.text)
                elif ev.kind == "token":
                    task.signals.token.emit(ev.text)
                elif ev.kind == "done":
                    final = ev.result
            return final

        def finished() -> None:
            self._ask_task = None
            self.chat.set_busy(False)

        t = self._run(fn, self._on_answer, on_error=self._on_ask_error, on_finished=finished)
        t.signals.status.connect(self.chat.set_status)
        t.signals.token.connect(self.chat.append_token)
        self._ask_task = t

    def _on_answer(self, res: AnswerResult | None) -> None:
        if res is None or self.paper is None:
            return
        text = res.text + ("\n\n*(stopped)*" if res.stopped else "")
        sources = [s.to_dict() for s in res.display_sources]
        self.chat.finish_assistant(text, sources, res.warnings)
        self.services.db.add_message(self.paper.document.doc_id, "assistant", text, sources)

    def _on_ask_error(self, exc: object) -> None:
        self.chat.finish_assistant("*The request failed.*", [], [])
        self._show_error(exc)

    def _on_stop(self) -> None:
        if self._ask_task:
            self._ask_task.cancel.set()
            self.chat.set_status("Stopping…")

    def _on_cite(self, msg_idx: int, sid: str) -> None:
        if self.paper is None:
            return
        try:
            src = next(s for s in self.chat._msgs[msg_idx].get("sources", []) if s["sid"] == sid)
        except (IndexError, StopIteration):
            if sid.startswith("H"):
                return
            self.status_lbl.setText(f"Source {sid} is not part of this answer's evidence list")
            return
        bbox = None
        if src.get("label"):
            item = self.paper.document.visual(src["label"])
            bbox = getattr(item, "bbox", None) or getattr(item, "caption_bbox", None)
        snippet = "" if bbox else src["excerpt"].lstrip("“\"").split("Referred to in the text")[0]
        self.viewer.show_evidence(src["page"], snippet, bbox)
        self.status_lbl.setText(f"Jumped to {src['title']}")

    def clear_chat(self) -> None:
        if self.paper is None:
            return
        self.services.db.clear_messages(self.paper.document.doc_id)
        if self.pipeline:
            self.pipeline.memory.clear()
        self.chat.clear()
        self.chat.add_notice("Chat cleared.")

    def export_chat(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export chat", "chat.md", "Markdown (*.md)")
        if path:
            Path(path).write_text(self.chat.transcript_markdown(), encoding="utf-8")

    # ------------------------------------------------------------------ models / settings
    def _check_backend(self) -> None:
        self.backend_lbl.setText("Model: checking…")
        self._run(lambda t: self.services.models.health(), self._on_health,
                  on_error=lambda e: self.backend_lbl.setText(f"Model: {e}"))

    def _on_health(self, res: tuple[bool, str]) -> None:
        ok, msg = res
        self.backend_lbl.setText(("● " if ok else "○ ") + msg)
        self.backend_lbl.setStyleSheet("color:#16a34a;" if ok else "color:#dc2626;")
        if not ok:
            self.chat.add_notice(f"<b>Model backend not ready:</b> {msg}", "error")

    def _unload(self) -> None:
        self.services.models.unload_all()
        self.status_lbl.setText("Models unloaded from memory")

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.services.cfg, self)
        if dlg.exec():
            path = save_user_config(dlg.patch())
            QMessageBox.information(self, "Settings saved",
                                    f"Saved to {path}.\nRestart Literature Buddy to apply the new models.")

    def about(self) -> None:
        QMessageBox.about(self, "About Literature Buddy",
                          f"<b>Literature Buddy {__version__}</b><br>Local, evidence-grounded reading assistant "
                          "for scientific papers.<br>Licensed under AGPL-3.0-or-later.")

    # ------------------------------------------------------------------ window events
    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # noqa: N802
        if e.mimeData().hasUrls() or e.mimeData().hasText():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:  # noqa: N802
        self.viewer.dropEvent(e)

    def closeEvent(self, e) -> None:  # noqa: N802
        for t in list(self._tasks):
            t.cancel.set()
        QThreadPool.globalInstance().waitForDone(3000)
        self.services.shutdown()
        e.accept()
