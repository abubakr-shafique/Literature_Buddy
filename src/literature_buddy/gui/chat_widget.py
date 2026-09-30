"""Chat panel: streaming answers, evidence colouring, clickable citations, thumbnails."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote, unquote

import markdown
from PySide6.QtCore import QTimer, QUrl, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import theme

SUGGESTIONS = [
    "What is the main contribution of this paper?",
    "What datasets and methods were used?",
    "What are the main results?",
    "What are the limitations of this study?",
]
_CITE = re.compile(r"\[([SH]\d+)\]")


class _Input(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        from PySide6.QtCore import Qt

        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class ChatWidget(QWidget):
    sendRequested = Signal(str)
    stopRequested = Signal()
    citationClicked = Signal(int, str)  # message index, source id

    def __init__(self) -> None:
        super().__init__()
        self._msgs: list[dict] = []
        self._image_root: Path | None = None
        self._busy = False
        self._status = ""
        self._dirty = False

        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._on_link)
        self.input = _Input()
        self.input.setPlaceholderText("Ask about this paper…  (Enter to send · Shift+Enter for a new line)")
        self.input.setFixedHeight(64)
        self.input.submitted.connect(self._submit)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._submit)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.stopRequested)
        self.stop_btn.hide()
        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("sectionlabel")

        row = QHBoxLayout()
        row.addWidget(self.input, 1)
        col = QVBoxLayout()
        col.addWidget(self.send_btn)
        col.addWidget(self.stop_btn)
        row.addLayout(col)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.status_lbl)
        lay.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._render)
        self.set_enabled(False)

    # ---- state ----
    def set_image_root(self, root: Path | None) -> None:
        self._image_root = root

    def set_enabled(self, enabled: bool) -> None:
        self.input.setEnabled(enabled and not self._busy)
        self.send_btn.setEnabled(enabled and not self._busy)
        self._enabled = enabled

    def set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = busy
        self.stop_btn.setVisible(busy)
        self.send_btn.setVisible(not busy)
        self.input.setEnabled(not busy and getattr(self, "_enabled", True))
        self.status_lbl.setText(status)
        if not busy:
            self.input.setFocus()

    def set_status(self, text: str) -> None:
        self.status_lbl.setText(text)

    # ---- messages ----
    def clear(self) -> None:
        self._msgs.clear()
        self._render()

    def add_user(self, text: str) -> None:
        self._msgs.append({"role": "user", "text": text})
        self._render()

    def add_notice(self, html: str, kind: str = "info") -> None:
        self._msgs.append({"role": "notice", "html": html, "kind": kind})
        self._render()

    def add_welcome(self, title: str, info: str) -> None:
        chips = "".join(
            f"<li><a href='lb://ask/{quote(q)}'>{q}</a></li>" for q in SUGGESTIONS)
        self.add_notice(f"<b>Ready:</b> {title}<br><span>{info}</span><br><br>Try:<ul>{chips}</ul>"
                        "Tip: drag over text in the paper to highlight it, then tick it as context.")

    def begin_assistant(self) -> None:
        self._msgs.append({"role": "assistant", "text": "", "sources": [], "warnings": [], "final": False})

    def append_token(self, piece: str) -> None:
        if self._msgs and self._msgs[-1]["role"] == "assistant":
            self._msgs[-1]["text"] += piece
            if not self._timer.isActive():
                self._timer.start()

    def finish_assistant(self, text: str, sources: list[dict], warnings: list[str]) -> None:
        m = self._msgs[-1]
        m.update(text=text, sources=sources, warnings=warnings, final=True)
        self._render()

    def load_history(self, messages: list[dict]) -> None:
        for m in messages:
            if m["role"] == "user":
                self._msgs.append({"role": "user", "text": m["content"]})
            else:
                self._msgs.append({"role": "assistant", "text": m["content"], "sources": m.get("sources", []),
                                   "warnings": [], "final": True})
        self._render()

    def transcript_markdown(self) -> str:
        out = []
        for m in self._msgs:
            if m["role"] == "user":
                out.append(f"**You:** {m['text']}\n")
            elif m["role"] == "assistant":
                out.append(f"**Literature Buddy:**\n\n{m['text']}\n")
                for s in m.get("sources", []):
                    out.append(f"- [{s['sid']}] {s['title']}: {s['excerpt']}")
                out.append("")
        return "\n".join(out)

    # ---- rendering ----
    def _submit(self) -> None:
        text = self.input.toPlainText().strip()
        if text and not self._busy and getattr(self, "_enabled", True):
            self.input.clear()
            self.sendRequested.emit(text)

    def _md(self, text: str, idx: int) -> str:
        safe = text.replace("&", "&amp;").replace("<", "&lt;")
        html = markdown.markdown(safe, extensions=["tables", "sane_lists", "nl2br"])
        c = theme.colors()
        for label, key in (("Paper states:", "stated"), ("Inference:", "infer"), ("General scientific context:", "general")):
            html = html.replace(f"<strong>{label}</strong>", f"<span style='color:{c[key]}; font-weight:700;'>{label}</span>")
        return _CITE.sub(lambda m: f"<a href='lb://cite/{idx}/{m.group(1)}'>[{m.group(1)}]</a>", html)

    def _bubble(self, html: str, bg: str, fg: str, right: bool) -> str:
        left, rgt = ("18%", "0") if right else ("0", "10%")
        return (f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
                f"<td width='{left}'></td><td bgcolor='{bg}' style='padding:9px 12px; color:{fg};'>{html}</td>"
                f"<td width='{rgt}'></td></tr></table><div style='font-size:4px;'>&nbsp;</div>")

    def _sources_html(self, m: dict, idx: int) -> str:
        c = theme.colors()
        if not m.get("sources"):
            return ""
        rows = []
        for s in m["sources"]:
            thumb = ""
            if s.get("image_path") and self._image_root:
                thumb = f"<br><img src='{(self._image_root / s['image_path']).as_posix()}' width='240'>"
            rows.append(
                f"<div style='margin-top:5px;'><a href='lb://cite/{idx}/{s['sid']}'>📄 [{s['sid']}] {s['title']}</a>"
                f"<br><span style='color:{c['muted']}; font-size:small;'>“{s['excerpt']}”</span>{thumb}</div>")
        return f"<hr><span style='color:{c['muted']}; font-size:small;'><b>Sources</b> (click to jump)</span>{''.join(rows)}"

    def _render(self) -> None:
        c = theme.colors()
        bar = self.view.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 30
        parts = [f"<body style='color:{c['text']}; font-size:11pt;'><style>a {{ color:{c['link']}; text-decoration:none; }}"
                 "table.md {border-collapse: collapse;} </style>"]
        for i, m in enumerate(self._msgs):
            if m["role"] == "user":
                t = m["text"].replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br>")
                parts.append(self._bubble(t, c["user_bg"], c["user_fg"], True))
            elif m["role"] == "notice":
                parts.append(self._bubble(m["html"], c["err_bg"] if m["kind"] == "error" else c["sys_bg"], c["text"], False))
            else:
                body = self._md(m["text"], i) if m["text"] else f"<i style='color:{c['muted']}'>Thinking…</i>"
                extra = ""
                if m.get("final"):
                    extra += self._sources_html(m, i)
                    for w in m.get("warnings", []):
                        extra += f"<div style='color:{c['infer']}; font-size:small;'>⚠ {w.replace('<', '&lt;')}</div>"
                parts.append(self._bubble(body + extra, c["bot_bg"], c["text"], False))
        parts.append("</body>")
        self.view.setHtml("".join(parts))
        if at_bottom or self._busy:
            bar.setValue(bar.maximum())

    def _on_link(self, url: QUrl) -> None:
        s = url.toString()
        if s.startswith("lb://ask/"):
            if not self._busy and getattr(self, "_enabled", True):
                self.sendRequested.emit(unquote(s[len("lb://ask/"):]))
        elif s.startswith("lb://cite/"):
            _, _, rest = s.partition("lb://cite/")
            idx, _, sid = rest.partition("/")
            self.citationClicked.emit(int(idx), sid)
