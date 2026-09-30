"""Background tasks: keep the GUI thread free while parsing, embedding and generating."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

from ..errors import Cancelled

log = logging.getLogger(__name__)


class TaskSignals(QObject):
    progress = Signal(float, str)
    status = Signal(str)
    token = Signal(str)
    result = Signal(object)
    error = Signal(object)
    cancelled = Signal()
    finished = Signal()


class Task(QRunnable):
    """Runs `fn(task)` in the global thread pool. `fn` may emit task.signals.* and must poll
    `task.cancel` (a threading.Event) in long loops."""

    def __init__(self, fn: Callable[[Task], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = TaskSignals()
        self.cancel = threading.Event()
        self.setAutoDelete(False)  # the owner keeps the reference until `finished`

    def run(self) -> None:
        try:
            self.signals.result.emit(self.fn(self))
        except Cancelled:
            self.signals.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            log.exception("Task failed")
            self.signals.error.emit(exc)
        finally:
            self.signals.finished.emit()
