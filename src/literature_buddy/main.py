"""Main entry point for Literature Buddy application."""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from .config.settings import load_settings, Settings
from .gui.main_window import MainWindow
from .models.model_loader import ModelLoader


def main():
    """Main application entry point."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Literature Buddy")
    app.setOrganizationName("LiteratureBuddy")
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Load settings
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    settings = load_settings(config_path)
    
    # Initialize model loader (load models from local directory)
    model_loader = ModelLoader(settings)
    
    # Create and show main window
    window = MainWindow(settings, model_loader)
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()