"""PDF viewer: page navigation, zoom, search, goto-page for citation links,
drag-and-drop, click-to-open, and annotation support."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import fitz
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (QDragEnterEvent, QDropEvent, QImage, QPixmap,
                           QPainter, QColor, QPen, QBrush, QMouseEvent)
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)


class Annotation:
    """Simple rectangle highlight/underline annotation."""

    def __init__(self, page: int, rect: QRectF, kind: str = "highlight"):
        self.page = page          # 1-based
        self.rect = rect          # in page coordinates (fitz)
        self.kind = kind          # "highlight" | "underline"
        self.text = ""            # extracted text for context


class PdfViewer(QWidget):
    pageChanged = Signal(int)
    pdfOpened = Signal(str)       # emitted with file path
    annotationAdded = Signal(object)  # Annotation

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._doc: Optional[fitz.Document] = None
        self._page = 0
        zoom = getattr(settings.ui, "pdf_default_zoom", 1.2) if settings else 1.2
        self._zoom = zoom
        self._annotations: list[Annotation] = []

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMouseTracking(True)
        self.image.mousePressEvent = self._on_mouse
        self.image.dragEnterEvent = self._drag_enter
        self.image.dropEvent = self._drop

        scroll = QScrollArea()
        scroll.setWidget(self.image)
        scroll.setWidgetResizable(True)

        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        self.zoom_out = QPushButton("−")
        self.zoom_in = QPushButton("+")
        self.highlight_btn = QPushButton("Highlight")
        self.underline_btn = QPushButton("Underline")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search text…")
        self.page_label = QLabel("No document")

        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)
        self.zoom_in.clicked.connect(lambda: self.set_zoom(self._zoom * 1.25))
        self.zoom_out.clicked.connect(lambda: self.set_zoom(self._zoom / 1.25))
        self.search.returnPressed.connect(self._do_search)
        self.highlight_btn.clicked.connect(lambda: self._start_selection("highlight"))
        self.underline_btn.clicked.connect(lambda: self._start_selection("underline"))

        self._selection_start: Optional[QPointF] = None
        self._selection_kind: Optional[str] = None

        bar = QHBoxLayout()
        for w in (self.prev_btn, self.page_label, self.next_btn,
                  self.zoom_out, self.zoom_in,
                  self.highlight_btn, self.underline_btn, self.search):
            bar.addWidget(w)
        lay = QVBoxLayout(self)
        lay.addWidget(scroll)
        lay.addLayout(bar)

    # -- API ---------------------------------------------------------------
    def open_pdf(self, path: str | Path) -> None:
        self._doc = fitz.open(str(path))
        self._page = 0
        self._annotations.clear()
        self._render()
        self.pdfOpened.emit(str(path))

    def goto_page(self, page_1_based: int) -> None:
        if self._doc is None:
            return
        self._page = max(0, min(int(page_1_based) - 1, len(self._doc) - 1))
        self._render()

    def prev_page(self):
        if self._doc:
            self.goto_page(max(1, self._page))

    def next_page(self):
        if self._doc:
            self.goto_page(min(len(self._doc), self._page + 2))

    def set_zoom(self, z: float):
        self._zoom = max(0.4, min(4.0, z))
        self._render()

    def _do_search(self):
        if self._doc is None or not self.search.text().strip():
            return
        q = self.search.text().lower()
        for i in range(self._page + 1, len(self._doc)):
            if q in self._doc[i].get_text().lower():
                self.goto_page(i + 1)
                return

    def _render(self) -> None:
        if self._doc is None:
            self.image.setPixmap(QPixmap())
            self.page_label.setText("No document")
            return
        page = self._doc[self._page]
        pix = page.get_pixmap(matrix=fitz.Matrix(self._zoom, self._zoom))
        img = QImage(pix.samples, pix.width, pix.height, pix.stride,
                     QImage.Format.Format_RGB888).copy()
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for ann in self._annotations:
            if ann.page != self._page + 1:
                continue
            rect = ann.rect
            # scale from page coords to pixmap coords
            r = QRectF(
                rect.x0 * self._zoom, rect.y0 * self._zoom,
                (rect.x1 - rect.x0) * self._zoom,
                (rect.y1 - rect.y0) * self._zoom,
            )
            if ann.kind == "highlight":
                painter.fillRect(r, QColor(255, 255, 0, 80))
            else:
                pen = QPen(QColor(255, 0, 0), 2)
                painter.setPen(pen)
                painter.drawLine(r.topLeft(), r.topRight())
        painter.end()
        self.image.setPixmap(QPixmap.fromImage(img))
        self.page_label.setText(f"Page {self._page + 1} of {len(self._doc)}")
        self.pageChanged.emit(self._page + 1)

    # -- drag & drop --------------------------------------------------------
    def _drag_enter(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(".pdf"):
                event.acceptProposedAction()

    def _drop(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(".pdf"):
                self.open_pdf(path)
                self.pdfOpened.emit(path)

    # -- click to open + annotation selection -------------------------------
    def _on_mouse(self, event: QMouseEvent):
        if self._doc is None:
            # click on empty viewer -> open dialog
            if event.button() == Qt.MouseButton.LeftButton:
                from PySide6.QtWidgets import QFileDialog
                path, _ = QFileDialog.getOpenFileName(
                    self, "Open paper", "", "PDF files (*.pdf)")
                if path:
                    self.open_pdf(path)
                    self.pdfOpened.emit(path)
            return

        if self._selection_kind:
            # finish selection
            pos = event.pos()
            if self._selection_start:
                x0 = min(self._selection_start.x(), pos.x()) / self._zoom
                y0 = min(self._selection_start.y(), pos.y()) / self._zoom
                x1 = max(self._selection_start.x(), pos.x()) / self._zoom
                y1 = max(self._selection_start.y(), pos.y()) / self._zoom
                rect = QRectF(x0, y0, x1 - x0, y1 - y0)
                if rect.width() > 5 and rect.height() > 5:
                    page = self._doc[self._page]
                    # convert to page coords for text extraction
                    fz_rect = fitz.Rect(x0, y0, x1, y1)
                    text = page.get_text("text", clip=fz_rect).strip()
                    ann = Annotation(page=self._page + 1,
                                     rect=fitz.Rect(x0, y0, x1, y1),
                                     kind=self._selection_kind)
                    ann.text = text
                    self._annotations.append(ann)
                    self._render()
                    self.annotationAdded.emit(ann)
            self._selection_start = None
            self._selection_kind = None
        else:
            # single click on blank area could also open file (optional)
            pass

    def _start_selection(self, kind: str):
        if self._doc is None:
            return
        self._selection_kind = kind
        # next mouse press will set start point
        self.image.mousePressEvent = lambda ev: self._selection_press(ev, kind)

    def _selection_press(self, event: QMouseEvent, kind: str):
        self._selection_start = QPointF(event.pos())
        # restore normal handler after release
        def release(ev):
            self._on_mouse(ev)
            self.image.mousePressEvent = self._on_mouse
        self.image.mouseReleaseEvent = release