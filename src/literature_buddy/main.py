"""Entry point: `literature-buddy` (see pyproject [project.scripts])."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from literature_buddy.config.settings import load_settings
from literature_buddy.gui.main_window import MainWindow


def main() -> int:
    settings = load_settings()
    app = QApplication(sys.argv)
    app.setApplicationName(settings.ui.window_title)
    win = MainWindow(settings)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())