"""List of user highlights with a check mark (= use as chat context) and clear buttons."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .types import Highlight


class HighlightsPanel(QWidget):
    toggled = Signal(str, bool)
    removed = Signal(list)  # list of highlight ids
    activated = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._items: dict[str, Highlight] = {}
        self._title = QLabel()
        self._title.setObjectName("sectionlabel")
        self._list = QListWidget()
        self._list.setToolTip("Tick a highlight to use it as context for your questions. Double-click to jump to it.")
        self._list.itemChanged.connect(self._on_changed)
        self._list.itemDoubleClicked.connect(lambda it: self.activated.emit(it.data(Qt.ItemDataRole.UserRole)))
        self._clear_sel = QPushButton("Clear selected")
        self._clear_sel.setObjectName("secondary")
        self._clear_all = QPushButton("Clear all")
        self._clear_all.setObjectName("secondary")
        self._clear_sel.clicked.connect(self._remove_selected)
        self._clear_all.clicked.connect(lambda: self.removed.emit(list(self._items)))
        row = QHBoxLayout()
        row.addWidget(self._clear_sel)
        row.addWidget(self._clear_all)
        row.addStretch()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 6)
        lay.addWidget(self._title)
        lay.addWidget(self._list)
        lay.addLayout(row)
        self._update_title()

    def _update_title(self) -> None:
        n = sum(1 for h in self._items.values() if h.use_in_chat)
        self._title.setText(f"Highlights ({len(self._items)}) — ☑ {n} used as chat context")
        self._clear_all.setEnabled(bool(self._items))
        self._clear_sel.setEnabled(bool(self._items))

    def _add_item(self, h: Highlight) -> None:
        text = " ".join(h.text.split())
        it = QListWidgetItem(f"p.{h.page + 1} · {text[:90]}{'…' if len(text) > 90 else ''}")
        it.setData(Qt.ItemDataRole.UserRole, h.hid)
        it.setToolTip(text[:600])
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Checked if h.use_in_chat else Qt.CheckState.Unchecked)
        self._list.blockSignals(True)
        self._list.addItem(it)
        self._list.blockSignals(False)

    def set_items(self, items: list[Highlight]) -> None:
        self._items = {h.hid: h for h in items}
        self._list.blockSignals(True)
        self._list.clear()
        self._list.blockSignals(False)
        for h in items:
            self._add_item(h)
        self._update_title()

    def add(self, h: Highlight) -> None:
        self._items[h.hid] = h
        self._add_item(h)
        self._update_title()

    def remove(self, hids: list[str]) -> None:
        for hid in hids:
            self._items.pop(hid, None)
        for i in reversed(range(self._list.count())):
            if self._list.item(i).data(Qt.ItemDataRole.UserRole) in hids:
                self._list.takeItem(i)
        self._update_title()

    def _on_changed(self, item: QListWidgetItem) -> None:
        hid = item.data(Qt.ItemDataRole.UserRole)
        use = item.checkState() == Qt.CheckState.Checked
        if hid in self._items:
            self._items[hid].use_in_chat = use
        self._update_title()
        self.toggled.emit(hid, use)

    def _remove_selected(self) -> None:
        ids = [i.data(Qt.ItemDataRole.UserRole) for i in self._list.selectedItems()]
        if ids:
            self.removed.emit(ids)

    def context(self) -> list[dict]:
        return [h.as_context() for h in self._items.values() if h.use_in_chat]

    def clear_all_items(self) -> None:
        self.set_items([])
