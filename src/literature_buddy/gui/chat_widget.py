"""Chat panel: message history (HTML with clickable citations), input box."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QHBoxLayout, QLineEdit, QPushButton, QScrollArea,
                               QTextBrowser, QVBoxLayout, QWidget)


class ChatWidget(QWidget):
    questionAsked = Signal(str)
    citationClicked = Signal(int)          # 1-based page

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        self.view.anchorClicked.connect(self._anchor)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask about this paper…")
        self.send_btn = QPushButton("Send")
        self.input.returnPressed.connect(self._send)
        self.send_btn.clicked.connect(self._send)

        row = QHBoxLayout(); row.addWidget(self.input); row.addWidget(self.send_btn)
        scroll = QScrollArea(); scroll.setWidget(self.view); scroll.setWidgetResizable(True)
        lay = QVBoxLayout(self); lay.addWidget(scroll); lay.addLayout(row)

    # rendering --------------------------------------------------------------
    def add_user_message(self, text: str) -> None:
        self.view.append(f"<b>You:</b> {text}")

    def start_assistant_message(self) -> None:
        self.view.append("<b>Assistant:</b> ")

    def append_assistant_chunk(self, text: str) -> None:
        cursor = self.view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)

    def finish_assistant_message(self, citations_html: str = "") -> None:
        if citations_html:
            self.view.append(f"<br><b>Sources:</b><br>{citations_html}")
        self.view.append("")

    def set_busy(self, busy: bool) -> None:
        self.input.setEnabled(not busy); self.send_btn.setEnabled(not busy)

    def _send(self) -> None:
        q = self.input.text().strip()
        if q:
            self.input.clear(); self.questionAsked.emit(q)

    def _anchor(self, url) -> None:
        if url.scheme() == "lb" and url.host() == "page":
            self.citationClicked.emit(int(url.path().lstrip("/")))