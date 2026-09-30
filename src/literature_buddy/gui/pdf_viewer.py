"""Continuous-scroll PDF viewer built on PyMuPDF rendering.

Features: lazy page rendering + LRU cache, zoom (buttons / Ctrl+wheel / fit width), text search,
drag-to-highlight with reading-order word selection, evidence overlay for citations,
click-to-open placeholder and drag & drop of PDF files or URLs.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict

import pymupdf
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMenu,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .types import Highlight, Rect

MARGIN = 14
_CACHE_PAGES = 14


class _PageWidget(QWidget):
    def __init__(self, viewer: PdfViewer, index: int) -> None:
        super().__init__()
        self.viewer, self.index = viewer, index
        self._drag_start: QPointF | None = None
        self._drag_now: QPointF | None = None
        self._live: list[Rect] = []
        self.setMouseTracking(False)
        self.setCursor(Qt.CursorShape.IBeamCursor)

    def _pdf(self, p: QPointF) -> tuple[float, float]:
        s = self.viewer.scale
        return p.x() / s, p.y() / s

    def paintEvent(self, _event) -> None:  # noqa: N802
        v = self.viewer
        p = QPainter(self)
        p.drawPixmap(0, 0, v.pixmap(self.index))
        s = v.scale

        def fill(rects, color: QColor) -> None:
            for r in rects:
                p.fillRect(QRectF(r[0] * s, r[1] * s, (r[2] - r[0]) * s, (r[3] - r[1]) * s), color)

        for h in v.highlights:
            if h.page == self.index:
                fill(h.rects, QColor(255, 214, 0, 105) if h.use_in_chat else QColor(160, 160, 160, 80))
        fill(v.state.search_hits.get(self.index, []), QColor(0, 170, 255, 110))
        fill(self._live, QColor(80, 120, 255, 90))
        ev = v.state.evidence.get(self.index, [])
        fill(ev, QColor(255, 120, 0, 90))
        p.setPen(QPen(QColor(255, 120, 0), 2))
        for r in ev:
            p.drawRect(QRectF(r[0] * s, r[1] * s, (r[2] - r[0]) * s, (r[3] - r[1]) * s))
        p.end()

    # -- selection --
    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = self._drag_now = e.position()
            self._live = []
            self.viewer.clear_evidence()
            self.update()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_now = e.position()
            self._live = self.viewer.select_words(self.index, self._pdf(self._drag_start), self._pdf(self._drag_now))[0]
            self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._drag_start is None:
            return
        a, b = self._pdf(self._drag_start), self._pdf(e.position())
        self._drag_start = self._drag_now = None
        self._live = []
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 4:  # a click is not a selection
            rects, text = self.viewer.select_words(self.index, a, b)
            if text.strip():
                self.viewer.selected.emit(self.index, rects, text)
        self.update()

    def contextMenuEvent(self, e) -> None:  # noqa: N802
        x, y = self._pdf(QPointF(e.pos()))
        hit = next((h for h in self.viewer.highlights if h.page == self.index
                    and any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in h.rects)), None)
        menu = QMenu(self)
        if hit:
            menu.addAction("Copy highlighted text", lambda: QApplication.clipboard().setText(hit.text))
            menu.addAction("Remove highlight", lambda: self.viewer.highlightRemoveRequested.emit(hit.hid))
        else:
            menu.addAction("Tip: drag over text to highlight it").setEnabled(False)
        menu.exec(QCursor.pos())


class _Scroll(QScrollArea):
    zoomRequested = Signal(int)

    def wheelEvent(self, e) -> None:  # noqa: N802
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoomRequested.emit(1 if e.angleDelta().y() > 0 else -1)
            e.accept()
        else:
            super().wheelEvent(e)


class _DropZone(QLabel):
    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("dropzone")
        self.setText(
            "<div style='font-size:20pt;'>📄</div>"
            "<div style='font-size:15pt; font-weight:600;'>Click to open a paper</div>"
            "<div style='font-size:11pt;'>or drag &amp; drop a PDF here<br>"
            "or use <b>Open URL</b> for arXiv, bioRxiv, PubMed Central…</div>")

    def mousePressEvent(self, _e) -> None:  # noqa: N802
        self.clicked.emit()


class PdfViewer(QWidget):
    pageChanged = Signal(int)
    selected = Signal(int, object, str)  # page, rects, text
    highlightRemoveRequested = Signal(str)
    fileDropped = Signal(str)
    urlDropped = Signal(str)
    openRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        from .types import ViewState

        self.state = ViewState()
        self.doc: pymupdf.Document | None = None
        self.highlights: list[Highlight] = []
        self._cache: OrderedDict[tuple[int, float], QPixmap] = OrderedDict()
        self._words: dict[int, list] = {}
        self._pages: list[_PageWidget] = []
        self._current = 0
        self._hits: list[tuple[int, Rect]] = []
        self._hit_i = -1
        self.setAcceptDrops(True)

        self._stack = QStackedWidget()
        self._drop = _DropZone()
        self._drop.clicked.connect(self.openRequested)
        self._scroll = _Scroll()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._scroll.setObjectName("pdfscroll")
        self._scroll.zoomRequested.connect(lambda d: self.set_scale(self.state.scale * (1.15 if d > 0 else 1 / 1.15)))
        self._scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self._container = QWidget()
        self._container.setObjectName("pdfcontainer")
        self._scroll.setWidget(self._container)
        self._stack.addWidget(self._drop)
        self._stack.addWidget(self._scroll)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._stack)

    # ---- properties ----
    @property
    def scale(self) -> float:
        return self.state.scale

    @property
    def page_count(self) -> int:
        return self.doc.page_count if self.doc else 0

    @property
    def current_page(self) -> int:
        return self._current

    # ---- document ----
    def open(self, path: str) -> None:
        """Load a PDF (raises on failure)."""
        doc = pymupdf.open(path)
        if doc.page_count == 0:
            raise ValueError("The PDF has no pages")
        self.close_document()
        self.doc = doc
        self.state.search_hits.clear()
        self.state.evidence.clear()
        self._hits, self._hit_i = [], -1
        self._pages = [_PageWidget(self, i) for i in range(doc.page_count)]
        for pw in self._pages:
            pw.setParent(self._container)
            pw.show()
        self._stack.setCurrentWidget(self._scroll)
        self.fit_width()
        self.goto(0)

    def close_document(self) -> None:
        for pw in self._pages:
            pw.deleteLater()
        self._pages, self._cache, self._words = [], OrderedDict(), {}
        if self.doc:
            self.doc.close()
        self.doc = None
        self.highlights = []
        self._stack.setCurrentWidget(self._drop)

    # ---- layout / rendering ----
    def _relayout(self) -> None:
        if not self.doc:
            return
        y, maxw = MARGIN, 0
        for i, pw in enumerate(self._pages):
            r = self.doc[i].rect
            w, h = int(r.width * self.scale), int(r.height * self.scale)
            pw.setGeometry(0, y, w, h)
            y += h + MARGIN
            maxw = max(maxw, w)
        vw = self._scroll.viewport().width()
        total_w = max(maxw + 2 * MARGIN, vw)
        for pw in self._pages:
            pw.move((total_w - pw.width()) // 2, pw.y())
        self._container.resize(total_w, y)

    def pixmap(self, idx: int) -> QPixmap:
        dpr = self.devicePixelRatioF()
        key = (idx, round(self.scale * dpr, 3))
        pm = self._cache.get(key)
        if pm is not None:
            self._cache.move_to_end(key)
            return pm
        z = self.scale * dpr
        pix = self.doc[idx].get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888).copy()
        pm = QPixmap.fromImage(img)
        pm.setDevicePixelRatio(dpr)
        self._cache[key] = pm
        while len(self._cache) > _CACHE_PAGES:
            self._cache.popitem(last=False)
        return pm

    def set_scale(self, s: float) -> None:
        s = max(0.3, min(s, 4.0))
        if not self.doc or abs(s - self.state.scale) < 1e-3:
            return
        anchor = self._current
        self.state.scale = s
        self._cache.clear()
        self._relayout()
        self.goto(anchor)
        for pw in self._pages:
            pw.update()

    def zoom_in(self) -> None:
        self.set_scale(self.scale * 1.15)

    def zoom_out(self) -> None:
        self.set_scale(self.scale / 1.15)

    def fit_width(self) -> None:
        if not self.doc:
            return
        widest = max(self.doc[i].rect.width for i in range(min(self.doc.page_count, 5)))
        self.set_scale((self._scroll.viewport().width() - 2 * MARGIN - 4) / widest)
        self._relayout()

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._relayout()

    # ---- navigation ----
    def goto(self, page: int, rect: Rect | None = None) -> None:
        if not self.doc or not self._pages:
            return
        page = max(0, min(page, self.page_count - 1))
        y = self._pages[page].y()
        if rect is not None:
            y += int(rect[1] * self.scale) - 60
        self._scroll.verticalScrollBar().setValue(max(y - 6, 0))
        self._set_current(page)

    def next_page(self) -> None:
        self.goto(self._current + 1)

    def prev_page(self) -> None:
        self.goto(self._current - 1)

    def _set_current(self, page: int) -> None:
        if page != self._current:
            self._current = page
            self.pageChanged.emit(page)

    def _on_scroll(self, value: int) -> None:
        if not self._pages:
            return
        probe = value + self._scroll.viewport().height() // 3
        cur = 0
        for i, pw in enumerate(self._pages):
            if pw.y() <= probe:
                cur = i
            else:
                break
        self._set_current(cur)

    # ---- text selection & highlights ----
    def words(self, idx: int) -> list:
        if idx not in self._words:
            self._words[idx] = self.doc[idx].get_text("words") if self.doc else []
        return self._words[idx]

    def select_words(self, idx: int, a: tuple[float, float], b: tuple[float, float]) -> tuple[list[Rect], str]:
        """Words between the two points in reading order -> (line rects, text)."""
        words = self.words(idx)
        if not words:
            return [], ""

        def nearest(pt: tuple[float, float]) -> int:
            best, bi = 1e18, 0
            for i, w in enumerate(words):
                dx = max(w[0] - pt[0], 0, pt[0] - w[2])
                dy = max(w[1] - pt[1], 0, pt[1] - w[3])
                d = dx * dx + 4 * dy * dy
                if d < best:
                    best, bi = d, i
            return bi

        lo, hi = sorted((nearest(a), nearest(b)))
        sel = words[lo: hi + 1]
        lines: dict[tuple[int, int], list[float]] = {}
        for w in sel:
            k = (w[5], w[6])
            r = lines.setdefault(k, [w[0], w[1], w[2], w[3]])
            r[0], r[1], r[2], r[3] = min(r[0], w[0]), min(r[1], w[1]), max(r[2], w[2]), max(r[3], w[3])
        return [tuple(r) for r in lines.values()], " ".join(w[4] for w in sel)  # type: ignore[misc]

    def add_highlight(self, h: Highlight) -> None:
        self.highlights.append(h)
        self._pages[h.page].update()

    def set_highlights(self, items: list[Highlight]) -> None:
        self.highlights = list(items)
        for pw in self._pages:
            pw.update()

    def refresh(self) -> None:
        for pw in self._pages:
            pw.update()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    # ---- search ----
    def search(self, text: str) -> int:
        self.state.search_hits.clear()
        self._hits, self._hit_i = [], -1
        if self.doc and text.strip():
            for i in range(self.doc.page_count):
                rects = [tuple(r) for r in self.doc[i].search_for(text.strip())]
                if rects:
                    self.state.search_hits[i] = rects  # type: ignore[assignment]
                    self._hits += [(i, r) for r in rects]  # type: ignore[misc]
        self.refresh()
        if self._hits:
            self.find_step(1)
        return len(self._hits)

    def find_step(self, d: int) -> None:
        if not self._hits:
            return
        self._hit_i = (self._hit_i + d) % len(self._hits)
        page, rect = self._hits[self._hit_i]
        self.goto(page, rect)

    # ---- evidence (citations) ----
    def clear_evidence(self) -> None:
        if self.state.evidence:
            self.state.evidence.clear()
            self.refresh()

    def show_evidence(self, page: int, snippet: str = "", bbox: Rect | None = None) -> None:
        """Navigate to a cited location and outline it (figure box or matching text)."""
        if not self.doc:
            return
        rects: list[Rect] = []
        if bbox is not None:
            rects = [bbox]
        elif snippet:
            words = " ".join(snippet.split()).split(" ")
            for n in (14, 8, 5):  # progressively shorter probes: search_for needs an exact run
                probe = " ".join(words[:n])
                found = self.doc[page].search_for(probe)
                if found:
                    rects = [tuple(r) for r in found]  # type: ignore[misc]
                    break
        self.state.evidence = {page: rects} if rects else {}
        self.refresh()
        self.goto(page, rects[0] if rects else None)

    # ---- drag & drop ----
    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls() or e.mimeData().hasText():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        md = e.mimeData()
        for url in md.urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(".pdf"):
                self.fileDropped.emit(url.toLocalFile())
                e.acceptProposedAction()
                return
            if url.scheme() in ("http", "https"):
                self.urlDropped.emit(url.toString())
                e.acceptProposedAction()
                return
        text = md.text().strip()
        if text.startswith(("http://", "https://")):
            self.urlDropped.emit(text)
            e.acceptProposedAction()
