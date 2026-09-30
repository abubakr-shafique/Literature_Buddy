"""Application stylesheet with automatic light/dark palettes."""

from __future__ import annotations

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

LIGHT = {"bg": "#f6f7f9", "panel": "#ffffff", "text": "#1f2937", "muted": "#6b7280", "border": "#e5e7eb",
         "accent": "#2563eb", "user_bg": "#2563eb", "user_fg": "#ffffff", "bot_bg": "#eef2f7",
         "sys_bg": "#fff7e6", "err_bg": "#fdecec", "paper_bg": "#dfe3e8", "link": "#1d4ed8",
         "stated": "#15803d", "infer": "#b45309", "general": "#7c3aed"}
DARK = {"bg": "#16181d", "panel": "#1e2128", "text": "#e5e7eb", "muted": "#9ca3af", "border": "#2f333c",
        "accent": "#60a5fa", "user_bg": "#1d4ed8", "user_fg": "#ffffff", "bot_bg": "#262a33",
        "sys_bg": "#3a3220", "err_bg": "#402424", "paper_bg": "#2a2d34", "link": "#93c5fd",
        "stated": "#4ade80", "infer": "#fbbf24", "general": "#c4b5fd"}


def is_dark() -> bool:
    return QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128


def colors() -> dict[str, str]:
    return DARK if is_dark() else LIGHT


def stylesheet() -> str:
    c = colors()
    return f"""
    QMainWindow, QDialog {{ background: {c['bg']}; }}
    QToolBar {{ background: {c['panel']}; border-bottom: 1px solid {c['border']}; spacing: 4px; padding: 3px; }}
    QToolButton {{ padding: 4px 8px; border-radius: 6px; }}
    QToolButton:hover {{ background: {c['bot_bg']}; }}
    QStatusBar {{ background: {c['panel']}; color: {c['muted']}; }}
    QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {{ background: {c['panel']}; border: 1px solid {c['border']};
        border-radius: 8px; padding: 5px 8px; color: {c['text']}; }}
    QPlainTextEdit:focus, QLineEdit:focus {{ border: 1px solid {c['accent']}; }}
    QTextBrowser {{ background: {c['panel']}; border: none; }}
    QPushButton {{ background: {c['accent']}; color: white; border: none; border-radius: 8px; padding: 7px 16px;
        font-weight: 600; }}
    QPushButton:disabled {{ background: {c['border']}; color: {c['muted']}; }}
    QPushButton#secondary {{ background: {c['bot_bg']}; color: {c['text']}; font-weight: 500; }}
    QPushButton#danger {{ background: #dc2626; }}
    QListWidget {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px; }}
    QListWidget::item {{ padding: 4px; }}
    QLabel#dropzone {{ background: {c['paper_bg']}; color: {c['muted']}; border: 2px dashed {c['border']};
        margin: 24px; border-radius: 18px; }}
    QLabel#dropzone:hover {{ border-color: {c['accent']}; color: {c['text']}; }}
    QScrollArea#pdfscroll, QWidget#pdfcontainer {{ background: {c['paper_bg']}; border: none; }}
    QSplitter::handle {{ background: {c['border']}; }}
    QLabel#sectionlabel {{ color: {c['muted']}; font-weight: 600; padding: 4px 2px; }}
    """
