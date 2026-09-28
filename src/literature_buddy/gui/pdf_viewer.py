"""PDF viewer: page navigation, zoom, search, goto-page for citation links."""

from __future__ import annotations

from pathlib import Path

import fitz
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)


class PdfViewer(QWidget):
    pageChanged = Signal(int)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._doc = None
        self._page = 0
        zoom = getattr(settings.ui, "pdf_default_zoom", 1.2) if settings else 1.2
        self._zoom = zoom

        self.image = QLabel(); self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll = QScrollArea(); scroll.setWidget(self.image)
        scroll.setWidgetResizable(True)

        self.prev_btn = QPushButton("◀"); self.next_btn = QPushButton("▶")
        self.zoom_out = QPushButton("−"); self.zoom_in = QPushButton("+")
        self.search = QLineEdit(); self.search.setPlaceholderText("Search text…")
        self.page_label = QLabel("No document")

        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)
        self.zoom_in.clicked.connect(lambda: self.set_zoom(self._zoom * 1.25))
        self.zoom_out.clicked.connect(lambda: self.set_zoom(self._zoom / 1.25))
        self.search.returnPressed.connect(self._do_search)

        bar = QHBoxLayout()
        for w in (self.prev_btn, self.page_label, self.next_btn,
                  self.zoom_out, self.zoom_in, self.search):
            bar.addWidget(w)
        lay = QVBoxLayout(self); lay.addWidget(scroll); lay.addLayout(bar)

    # -- API ---------------------------------------------------------------
    def open_pdf(self, path: str | Path) -> None:
        self._doc = fitz.open(str(path)); self._page = 0
        self._render()

    def goto_page(self, page_1_based: int) -> None:
        if self._doc is None: return
        self._page = max(0, min(int(page_1_based) - 1, len(self._doc) - 1))
        self._render()

    def prev_page(self): self.goto_page(self._page)  # _page is 0-based
    def next_page(self): self.goto_page(self._page + 2)
    def set_zoom(self, z: float): self._zoom = max(0.4, min(4.0, z)); self._render()

    def _do_search(self):
        if self._doc is None or not self.search.text().strip(): return
        for i in range(self._page + 1, len(self._doc)):
            if self.search.text().lower() in self._doc[i].get_text().lower():
                self.goto_page(i + 1); return

    def _render(self) -> None:
        if self._doc is None: return
        page = self._doc[self._page]
        pix = page.get_pixmap(matrix=fitz.Matrix(self._zoom, self._zoom))
        img = QImage(pix.samples, pix.width, pix.height, pix.stride,
                     QImage.Format.Format_RGB888).copy()
        self.image.setPixmap(QPixmap.fromImage(img))
        self.page_label.setText(f"Page {self._page + 1} of {len(self._doc)}")
        self.pageChanged.emit(self._page + 1)